import numpy as np

from qtpy import QtCore
from qtpy import QtGui
from qtpy import QtWidgets

from aidia import CFONT
from aidia.shape import Shape
from aidia.qt import distance
from aidia.image import gamma_correct, change_contrast, dicom_transform, graylevel_transform


CURSOR_DEFAULT = QtCore.Qt.ArrowCursor
CURSOR_POINT = QtCore.Qt.PointingHandCursor
CURSOR_DRAW = QtCore.Qt.CrossCursor
CURSOR_MOVE = QtCore.Qt.ClosedHandCursor
CURSOR_GRAB = QtCore.Qt.OpenHandCursor


class Canvas(QtWidgets.QWidget):

    zoomRequest = QtCore.Signal(int, QtCore.QPoint)
    scrollRequest = QtCore.Signal(int, int)
    fileOpenRequest = QtCore.Signal(int)
    newShape = QtCore.Signal()
    selectionChanged = QtCore.Signal(list)
    shapeMoved = QtCore.Signal()
    drawingPolygon = QtCore.Signal(bool)
    edgeSelected = QtCore.Signal(bool, object)
    vertexSelected = QtCore.Signal(bool)

    shapeDoubleClicked = QtCore.Signal()
    setDirty = QtCore.Signal()
    updateStatus = QtCore.Signal(str)

    CREATE, EDIT = 0, 1

    BR_STEP = 200
    CO_STEP = 200

    WC_STEP = 1
    WW_STEP = 1

    _createMode = "polygon"
    _fill_drawing = False

    def __init__(self, *args, **kwargs):
        self.epsilon = kwargs.pop('epsilon', 10.0)

        super().__init__(*args, **kwargs)
        # Initialise local state.
        self.mode = self.EDIT
        self.shapes = []
        self.shapesBackups = []
        self.current = None
        self.selectedShapes = []  # save the selected shapes here
        self.selectedShapesCopy = []
        # self.line represents:
        #   - createMode == 'polygon': edge from last point to current
        #   - createMode == 'rectangle': diagonal line of the rectangle
        self.line = Shape()
        self.prevPoint = QtCore.QPoint()
        self.prevMovePoint = QtCore.QPoint()
        self.offsets = QtCore.QPoint(), QtCore.QPoint()
        self.scale = 1.0
        self.image = QtGui.QImage()
        self.src_image = np.empty((0, 0))
        self.img_array = None
        self.pixmap = QtGui.QPixmap()
        self.bits = None  # bits allocated value in DICOM
        self.brightness = 0.0
        self.contrast = 1.0
        self.wc = self.original_wc = 0
        self.ww = self.original_ww = 0
        self.is_dicom = False
        self.pixel_spacing = 0
        self.diff = None
        self.prev_pos = None
        self.moving_distance = 0.0
        self.visible = {}
        self._hideBackround = False
        self.hideBackround = False
        self.hShape = None
        self.prevhShape = None
        self.hVertex = None
        self.prevhVertex = None
        self.hEdge = None
        self.prevhEdge = None
        self.movingShape = False
        self.paint_polygon = True
        self._painter = QtGui.QPainter()
        self.is_show_label = True
        self._cursor = CURSOR_DEFAULT
        self.menus = QtWidgets.QMenu()

        # self.mff = None
        self.target_label = None

        # Set widget options.
        self.setMouseTracking(True)
        self.setFocusPolicy(QtCore.Qt.WheelFocus)


    def fillDrawing(self):
        return self._fill_drawing

    def setFillDrawing(self, value):
        self._fill_drawing = value

    @property
    def createMode(self):
        return self._createMode

    @createMode.setter
    def createMode(self, value):
        if value not in [
            "polygon",
            "rectangle",
            "circle",
            "linestrip",
            "line",
            "point",
        ]:
            raise ValueError(f"Unsupported createMode: {value}")
        self._createMode = value

    def storeShapes(self):
        shapesBackup = []
        for shape in self.shapes:
            shapesBackup.append(shape.copy())
        if len(self.shapesBackups) >= 10:
            self.shapesBackups = self.shapesBackups[-9:]
        self.shapesBackups.append(shapesBackup)

    @property
    def isShapeRestorable(self):
        if len(self.shapesBackups) < 2:
            return False
        return True

    def restoreShape(self):
        if not self.isShapeRestorable:
            return
        self.shapesBackups.pop()  # latest
        shapesBackup = self.shapesBackups.pop()
        self.shapes = shapesBackup
        self.selectedShapes = []
        for shape in self.shapes:
            shape.selected = False
        self.update()

    def enterEvent(self, ev):
        self.overrideCursor(self._cursor)

    def leaveEvent(self, ev):
        self.unHighlight()
        self.restoreCursor()

    def focusOutEvent(self, ev):
        self.restoreCursor()

    def isVisible(self, shape):
        return self.visible.get(shape, True)

    def drawing(self):
        return self.mode == self.CREATE

    def editing(self):
        return self.mode == self.EDIT


    def setEditing(self, value=True):
        self.mode = self.EDIT if value else self.CREATE
        if not value:  # Create
            self.unHighlight()
            self.deSelectShape()
   

    def unHighlight(self):
        if self.hShape:
            self.hShape.highlightClear()
            self.update()
        self.prevhShape = self.hShape
        self.prevhVertex = self.hVertex
        self.prevhEdge = self.hEdge
        self.hShape = self.hVertex = self.hEdge = None

    def selectedVertex(self):
        return self.hVertex is not None

    def selectedEdge(self):
        return self.hEdge is not None

    def mouseMoveEvent(self, ev):
        """Update line with last point and current coordinates."""
        try:
            pos = self.transformPos(ev.localPos())
        except AttributeError:
            return

        self.prevMovePoint = pos

        # Update window center and window width.
        if QtCore.Qt.RightButton & ev.buttons():
            c = int((ev.localPos().y() - self.prev_pos.y()))
            w = int((ev.localPos().x() - self.prev_pos.x()))
            self.prev_pos = ev.localPos()
            self.moving_distance += np.sqrt(c**2 + w**2)
            if self.is_dicom:
                self.wc = int(self.wc + (c / self.WC_STEP))
                ww = int(self.ww - (w / self.WW_STEP))
                self.ww = ww if ww > 0 else self.ww
            else:
                brightness = self.brightness - (c / self.BR_STEP)
                contrast = self.contrast + (w / self.CO_STEP)
                self.brightness = brightness if -1.0 < brightness < 1.0 else self.brightness
                self.contrast = contrast if 0.0 < contrast < 2.0 else self.contrast
            self.update_image()
            self.paint_polygon = False
        
        # left click scroll
        if (QtCore.Qt.LeftButton & ev.buttons() and
            self.editing() and not self.selectedVertex()):
            diff_x, diff_y = self.diff
            diff_x += int((ev.localPos().x() - self.prev_pos.x()))
            diff_y += int((ev.localPos().y() - self.prev_pos.y()))
            self.prev_pos = ev.localPos()
            self.scrollRequest.emit(diff_x, QtCore.Qt.Horizontal)
            self.scrollRequest.emit(diff_y, QtCore.Qt.Vertical)
            self.diff = (diff_x, diff_y)
            # self.paint_polygon = False
            # self.update()

        # Polygon drawing.
        if self.drawing():
            self.line.shape_type = self.createMode

            if not self.current:
                self.overrideCursor(CURSOR_DRAW)
                return

            if self.outOfPixmap(pos):
                # Don't allow the user to draw outside the pixmap.
                # Project the point to the pixmap's edges.
                pos = self.intersectionPoint(self.current[-1], pos)
            elif (
                len(self.current) > 1
                and self.createMode == "polygon"
                and self.closeEnough(pos, self.current[0])
            ):
                # Attract line to starting point and
                # colorise to alert the user.
                pos = self.current[0]
                self.overrideCursor(CURSOR_POINT)
                self.current.highlightVertex(0, Shape.NEAR_VERTEX)
            else:
                self.overrideCursor(CURSOR_DRAW)

            if self.createMode in ["polygon", "linestrip"]:
                self.line[0] = self.current[-1]
                self.line[1] = pos
            elif self.createMode == "rectangle":
                self.line.points = [self.current[0], pos]
                self.line.close()
            elif self.createMode == "line":
                self.line.points = [self.current[0], pos]
                self.line.close()
            elif self.createMode == "point":
                self.line.points = [self.current[0]]
                self.line.close()
            self.update()
            self.current.highlightClear()
            return

       # Polygon or Vertex moving.
        if QtCore.Qt.LeftButton & ev.buttons():
            if self.selectedVertex():
                self.boundedMoveVertex(pos)
                self.update()
                self.movingShape = True
            return

        for shape in reversed([s for s in self.shapes if self.isVisible(s)]):
            # Look for a nearby vertex to highlight. If that fails,
            # check if we happen to be inside a shape.
            index = shape.nearestVertex(pos, self.epsilon / self.scale)
            index_edge = shape.nearestEdge(pos, self.epsilon / self.scale)
            if index is not None:
                if self.selectedVertex():
                    self.hShape.highlightClear()
                self.prevhVertex = self.hVertex = index
                self.prevhShape = self.hShape = shape
                self.prevhEdge = self.hEdge
                self.hEdge = None
                shape.highlightVertex(index, shape.MOVE_VERTEX)
                self.overrideCursor(CURSOR_POINT)
                self.update()
                break
            elif index_edge is not None and shape.canAddPoint():
                if self.selectedVertex():
                    self.hShape.highlightClear()
                self.prevhVertex = self.hVertex
                self.hVertex = None
                self.prevhShape = self.hShape = shape
                self.prevhEdge = self.hEdge = index_edge
                self.overrideCursor(CURSOR_POINT)
                self.update()
                break
            elif shape.containsPoint(pos):
                if self.selectedVertex():
                    self.hShape.highlightClear()
                self.prevhVertex = self.hVertex
                self.hVertex = None
                self.prevhShape = self.hShape = shape
                self.prevhEdge = self.hEdge
                self.hEdge = None
                self.overrideCursor(CURSOR_DEFAULT)
                self.update()
                break
        else:  # Nothing found, clear highlights, reset state.
            self.unHighlight()
            self.restoreCursor()
        self.vertexSelected.emit(self.hVertex is not None)

        # Update status bar.
        r, g, b = -1, -1, -1
        if 0 <= pos.x() <= self.pixmap.width() and 0 <= pos.y() <= self.pixmap.height():
            pixel = self.image.pixelColor(pos.x(), pos.y())
            r, g, b = (pixel.red(), pixel.green(), pixel.blue())
        if not self.is_dicom:
            txt = self.tr('pos({}, {}), value: ({:5d}, {:5d}, {:5d}), brightness: {:.4f}, contrast: {:.4f}').format(
                    int(pos.x()), int(pos.y()), r, g, b, self.brightness, self.contrast)
            self.updateStatus.emit(txt)
        elif self.is_dicom and self.wc and self.ww:
            x = (pos.x() - self.pixmap.width() // 2) * self.pixel_spacing
            y = (pos.y() - self.pixmap.height() // 2) * self.pixel_spacing
            txt = self.tr('pos: ({:4d}, {:4d}), value: ({:5d}, {:5d}, {:5d}), window center: {:6d}, window width: {:6d}').format(
                    int(x), int(y), r, g, b, int(self.wc), int(self.ww))
            self.updateStatus.emit(txt)
                

    def addPointToEdge(self):
        shape = self.prevhShape
        index = self.prevhEdge
        point = self.prevMovePoint
        if shape is None or index is None or point is None:
            return
        shape.insertPoint(index, point)
        shape.highlightVertex(index, shape.MOVE_VERTEX)
        self.hShape = shape
        self.hVertex = index
        self.hEdge = None
        self.movingShape = True

    def removeSelectedPoint(self):
        shape = self.prevhShape
        point = self.prevMovePoint
        if shape is None or point is None:
            return
        index = shape.nearestVertex(point, self.epsilon)
        if index is None:
            return
        shape.removePoint(index)
        # shape.highlightVertex(index, shape.MOVE_VERTEX)
        self.hShape = shape
        self.hVertex = None
        self.hEdge = None
        self.movingShape = True  # Save changes

    def mousePressEvent(self, ev):
        pos = self.transformPos(ev.localPos())
        self.prev_pos = ev.localPos()  # for contrast adjustment
        self.moving_distance = 0.0
        self.diff = (0, 0)
        if ev.button() == QtCore.Qt.LeftButton:
            if self.drawing():
                if self.current:
                    if self.createMode == "polygon":
                        self.current.addPoint(self.line[1])
                        self.line[0] = self.current[-1]
                        if self.current.isClosed():
                            self.finalise()
                    elif self.createMode in ["rectangle", "line"]:
                        assert len(self.current.points) == 1
                        self.current.points = self.line.points
                        self.finalise()
                    elif self.createMode == "linestrip":
                        self.current.addPoint(self.line[1])
                        self.line[0] = self.current[-1]
                        if int(ev.modifiers()) == QtCore.Qt.ControlModifier:
                            self.finalise()
                elif not self.outOfPixmap(pos):
                    # Create new shape.
                    self.current = Shape(shape_type=self.createMode)
                    self.current.addPoint(pos)
                    if self.createMode == "point":
                        self.finalise()
                    else:
                        self.line.points = [pos, pos]
                        self.setHiding()
                        self.drawingPolygon.emit(True)
                        self.update()
            elif self.editing():
                if self.selectedEdge():
                    self.addPointToEdge()
                elif (self.selectedVertex() and int(ev.modifiers()) == QtCore.Qt.ShiftModifier):
                    # Delete point if: left-click + SHIFT on a point
                    self.removeSelectedPoint()

                group_mode = int(ev.modifiers()) == QtCore.Qt.ControlModifier
                self.selectShapePoint(pos, multiple_selection_mode=group_mode)
                self.prevPoint = pos
                self.update()
            else:
                group_mode = (int(ev.modifiers()) == QtCore.Qt.ControlModifier)
                self.selectShapePoint(pos, multiple_selection_mode=group_mode)
                self.prevPoint = pos
                self.update()
        elif ev.button() == QtCore.Qt.RightButton and self.editing():
            group_mode = int(ev.modifiers()) == QtCore.Qt.ControlModifier
            if not self.selectedShapes or (
                self.hShape is not None
                and self.hShape not in self.selectedShapes
            ):
                self.selectShapePoint(pos, multiple_selection_mode=group_mode)
                self.update()
            self.prevPoint = pos
            # if self.editing():
            #     group_mode = (int(ev.modifiers()) == QtCore.Qt.ControlModifier)
            #     self.selectShapePoint(pos, multiple_selection_mode=group_mode)
            #     self.prevPoint = pos
            #     self.update()


    def mouseReleaseEvent(self, ev):
        self.paint_polygon = True
        self.update()

        if self.moving_distance > 10:
            # if ev.button() == QtCore.Qt.RightButton and self.is_dicom_image():
            if ev.button() == QtCore.Qt.RightButton:
                # self.setDirty.emit()
                self.update()

        # Popup right click menus.
        elif ev.button() == QtCore.Qt.RightButton:
            menu = self.menus
            self.restoreCursor()
            if not menu.exec_(self.mapToGlobal(ev.pos())) and self.selectedShapesCopy:
                # Cancel the move by deleting the shadow copy.
                self.selectedShapesCopy = []
                self.update()
        
        if self.movingShape and self.hShape:
            index = self.shapes.index(self.hShape)
            if self.shapesBackups[-1][index].points != self.shapes[index].points:
                self.storeShapes()
                self.shapeMoved.emit()

            self.movingShape = False
        

    def endMove(self, copy):
        assert self.selectedShapes and self.selectedShapesCopy
        assert len(self.selectedShapesCopy) == len(self.selectedShapes)
        if copy:
            for i, shape in enumerate(self.selectedShapesCopy):
                self.shapes.append(shape)
                self.selectedShapes[i].selected = False
                self.selectedShapes[i] = shape
        else:
            for i, shape in enumerate(self.selectedShapesCopy):
                self.selectedShapes[i].points = shape.points
        self.selectedShapesCopy = []
        self.update()
        self.storeShapes()
        return True

    def hideBackroundShapes(self, value):
        self.hideBackround = value
        if self.selectedShapes:
            # Only hide other shapes if there is a current selection.
            # Otherwise the user will not be able to select a shape.
            self.setHiding(True)
            self.update()

    def setHiding(self, enable=True):
        self._hideBackround = self.hideBackround if enable else False

    def canCloseShape(self):
        return self.drawing() and self.current and len(self.current) > 2

    def mouseDoubleClickEvent(self, ev):
        pos = self.transformPos(ev.localPos())
        # We need at least 4 points here, since the mousePress handler
        # adds an extra one before this handler is called.
        if self.canCloseShape() and len(self.current) > 3:
            self.current.popPoint()
            self.finalise()
        elif self.editing():
            for shape in reversed(self.shapes):
                if self.isVisible(shape) and shape.containsPoint(pos):
                    self.shapeDoubleClicked.emit()

    def selectShapes(self, shapes):
        self.setHiding()
        self.selectionChanged.emit(shapes)
        self.update()

    def selectShapePoint(self, point, multiple_selection_mode):
        """Select the first shape created which contains this point."""
        if self.selectedVertex():  # A vertex is marked for selection.
            index, shape = self.hVertex, self.hShape
            shape.highlightVertex(index, shape.MOVE_VERTEX)
        else:
            for shape in reversed(self.shapes):
                if self.isVisible(shape) and shape.containsPoint(point):
                    self.calculateOffsets(shape, point)
                    self.setHiding()
                    if multiple_selection_mode:
                        if shape not in self.selectedShapes:
                            self.selectionChanged.emit(
                                self.selectedShapes + [shape])
                    else:
                        self.selectionChanged.emit([shape])
                    return
        self.deSelectShape()

    def calculateOffsets(self, shape, point):
        rect = shape.boundingRect()
        x1 = rect.x() - point.x()
        y1 = rect.y() - point.y()
        x2 = (rect.x() + rect.width() - 1) - point.x()
        y2 = (rect.y() + rect.height() - 1) - point.y()
        self.offsets = QtCore.QPoint(x1, y1), QtCore.QPoint(x2, y2)

    def boundedMoveVertex(self, pos):
        index, shape = self.hVertex, self.hShape
        point = shape[index]
        if self.outOfPixmap(pos):
            pos = self.intersectionPoint(point, pos)
        shape.moveVertexBy(index, pos - point)

    def boundedMoveShapes(self, shapes, pos):
        if self.outOfPixmap(pos):
            return False  # No need to move
        o1 = pos + self.offsets[0]
        if self.outOfPixmap(o1):
            pos -= QtCore.QPoint(min(0, o1.x()), min(0, o1.y()))
        o2 = pos + self.offsets[1]
        if self.outOfPixmap(o2):
            pos += QtCore.QPoint(min(0, self.pixmap.width() - o2.x()),
                                 min(0, self.pixmap.height() - o2.y()))
        # XXX: The next line tracks the new position of the cursor
        # relative to the shape, but also results in making it
        # a bit "shaky" when nearing the border and allows it to
        # go outside of the shape's area for some reason.
        # self.calculateOffsets(self.selectedShapes, pos)
        dp = pos - self.prevPoint
        if dp:
            for shape in shapes:
                shape.moveBy(dp)
            self.prevPoint = pos
            return True
        return False

    def deSelectShape(self):
        if self.selectedShapes:
            self.setHiding(False)
            self.selectionChanged.emit([])
            self.update()

    def deleteSelected(self):
        deleted_shapes = []
        if self.selectedShapes:
            for shape in self.selectedShapes:
                self.shapes.remove(shape)
                deleted_shapes.append(shape)
            self.storeShapes()
            self.selectedShapes = []
            self.update()
        return deleted_shapes

    def copySelectedShapes(self):
        if self.selectedShapes:
            self.selectedShapesCopy = [s.copy() for s in self.selectedShapes]
            self.boundedShiftShapes(self.selectedShapesCopy)
            self.endMove(copy=True)
        return self.selectedShapes

    def boundedShiftShapes(self, shapes):
        # Try to move in one direction, and if it fails in another.
        # Give up if both fail.
        point = shapes[0][0]
        offset = QtCore.QPoint(2.0, 2.0)
        self.offsets = QtCore.QPoint(), QtCore.QPoint()
        self.prevPoint = point
        if not self.boundedMoveShapes(shapes, point - offset):
            self.boundedMoveShapes(shapes, point + offset)

    # main paint event
    def paintEvent(self, event):
        if not self.pixmap:
            return super().paintEvent(event)

        p = self._painter
        p.begin(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.setRenderHint(QtGui.QPainter.HighQualityAntialiasing)
        p.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)

        p.scale(self.scale, self.scale)
        p.translate(self.offsetToCenter())

        text_size = 14
        font = QtGui.QFont("Arial", text_size, QtGui.QFont.Bold, False)
        fm = QtGui.QFontMetrics(font)
        text_height = fm.height()

        p.drawPixmap(0, 0, self.pixmap)
        Shape.scale = self.scale

        if self.paint_polygon:
            for shape in self.shapes:
                if (shape.selected or not self._hideBackround) and self.isVisible(shape):
                    strings = str(shape.label).split('_')
                    # shape.fill = shape.selected or shape == self.hShape
                    # if self.target_label is None or target_in_list(self.target_label, strings):
                    shape.paint(p)

                    # Calculate center cordinates.
                    if self.is_show_label:
                        x_list = [point.x() for point in shape.points]
                        y_list = [point.y() for point in shape.points]
                        x = sum(x_list) / len(x_list) - text_size / 2
                        y = max(y_list)
                        # Display labels.
                        label_point = QtCore.QPoint(int(x), int(y))
                        p.setFont(font)
                        p.setPen(shape.label_color)
                        for s in strings:
                            label_point.setX(label_point.x())
                            label_point.setY(label_point.y() + text_height)
                            p.drawText(label_point, s)
        if self.current:
            self.current.paint(p)
            self.line.paint(p)
        if self.selectedShapesCopy:
            for s in self.selectedShapesCopy:
                s.paint(p)

        if (
                self.fillDrawing()
                and self.createMode == 'polygon'
                and self.current is not None
                and len(self.current.points) >= 2
        ):
            drawing_shape = self.current.copy()
            drawing_shape.addPoint(self.line[1])
            drawing_shape.fill = True
            # drawing_shape.fill_color.setAlpha(64)
            drawing_shape.paint(p)

        p.end()

    def transformPos(self, point):
        """Convert from widget-logical coordinates to painter-logical ones."""
        return point / self.scale - self.offsetToCenter()

    def offsetToCenter(self):
        s = self.scale
        area = super(Canvas, self).size()
        w, h = self.pixmap.width() * s, self.pixmap.height() * s
        aw, ah = area.width(), area.height()
        x = (aw - w) / (2 * s) if aw > w else 0
        y = (ah - h) / (2 * s) if ah > h else 0
        return QtCore.QPoint(x, y)

    def outOfPixmap(self, p):
        w, h = self.pixmap.width(), self.pixmap.height()
        return not (0 <= p.x() <= w - 1 and 0 <= p.y() <= h - 1)

    def finalise(self):
        assert self.current
        self.current.close()
        self.shapes.append(self.current)
        self.storeShapes()
        self.current = None
        self.setHiding(False)
        self.newShape.emit()
        self.update()

    def closeEnough(self, p1, p2):
        # d = distance(p1 - p2)
        # m = (p1-p2).manhattanLength()
        # print "d %.2f, m %d, %.2f" % (d, m, d - m)
        # divide by scale to allow more precision when zoomed in
        return distance(p1 - p2) < (self.epsilon / self.scale)

    def intersectionPoint(self, p1, p2):
        # Cycle through each image edge in clockwise fashion,
        # and find the one intersecting the current line segment.
        # http://paulbourke.net/geometry/lineline2d/
        size = self.pixmap.size()
        points = [(0, 0),
                  (size.width() - 1, 0),
                  (size.width() - 1, size.height() - 1),
                  (0, size.height() - 1)]
        # x1, y1 should be in the pixmap, x2, y2 should be out of the pixmap
        x1 = min(max(p1.x(), 0), size.width() - 1)
        y1 = min(max(p1.y(), 0), size.height() - 1)
        x2, y2 = p2.x(), p2.y()
        d, i, (x, y) = min(self.intersectingEdges((x1, y1), (x2, y2), points))
        x3, y3 = points[i]
        x4, y4 = points[(i + 1) % 4]
        if (x, y) == (x1, y1):
            # Handle cases where previous point is on one of the edges.
            if x3 == x4:
                return QtCore.QPoint(x3, min(max(0, y2), max(y3, y4)))
            else:  # y3 == y4
                return QtCore.QPoint(min(max(0, x2), max(x3, x4)), y3)
        return QtCore.QPoint(x, y)

    def intersectingEdges(self, point1, point2, points):
        """Find intersecting edges.

        For each edge formed by `points', yield the intersection
        with the line segment `(x1,y1) - (x2,y2)`, if it exists.
        Also return the distance of `(x2,y2)' to the middle of the
        edge along with its index, so that the one closest can be chosen.
        """
        (x1, y1) = point1
        (x2, y2) = point2
        for i in range(4):
            x3, y3 = points[i]
            x4, y4 = points[(i + 1) % 4]
            denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
            nua = (x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)
            nub = (x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)
            if denom == 0:
                # This covers two cases:
                #   nua == nub == 0: Coincident
                #   otherwise: Parallel
                continue
            ua, ub = nua / denom, nub / denom
            if 0 <= ua <= 1 and 0 <= ub <= 1:
                x = x1 + ua * (x2 - x1)
                y = y1 + ua * (y2 - y1)
                m = QtCore.QPoint((x3 + x4) / 2, (y3 + y4) / 2)
                d = distance(m - QtCore.QPoint(x2, y2))
                yield d, i, (x, y)

    # These two, along with a call to adjustSize are required for the
    # scroll area.
    def sizeHint(self):
        return self.minimumSizeHint()

    def minimumSizeHint(self):
        if self.pixmap:
            return self.scale * self.pixmap.size()
        return super(Canvas, self).minimumSizeHint()

    def wheelEvent(self, ev):
        mods = ev.modifiers()
        delta = ev.angleDelta()
        if QtCore.Qt.AltModifier == int(mods):
            self.fileOpenRequest.emit(delta.x())
        elif QtCore.Qt.ControlModifier == int(mods):
            self.zoomRequest.emit(delta.y(), ev.pos())
        else:
            if QtCore.Qt.ShiftModifier == int(mods):
                self.scrollRequest.emit(delta.y(), QtCore.Qt.Horizontal)
            else:
                self.scrollRequest.emit(delta.y(), QtCore.Qt.Vertical)
        ev.accept()

    def keyPressEvent(self, ev):
        key = ev.key()
        if key == QtCore.Qt.Key_Escape and self.current:
            self.current = None
            self.drawingPolygon.emit(False)
            self.update()
        elif key == QtCore.Qt.Key_Return and self.canCloseShape():
            self.finalise()

    def setLastLabel(self, text):
        # assert text
        self.shapes[-1].label = text
        self.shapesBackups.pop()
        self.storeShapes()
        return self.shapes[-1]

    def undoLastLine(self):
        assert self.shapes
        self.current = self.shapes.pop()
        self.current.setOpen()
        # self.line.points = [self.current[-1], self.current[0]]
        if self.createMode in ["polygon", "linestrip"]:
            self.line.points = [self.current[-1], self.current[0]]
        elif self.createMode in ["rectangle", "line"]:
            self.current.points = self.current.points[0:1]
        elif self.createMode == "point":
            self.current = None
        self.drawingPolygon.emit(True)

    def undoLastPoint(self):
        if not self.current or self.current.isClosed():
            return
        self.current.popPoint()
        if len(self.current) > 0:
            self.line[0] = self.current[-1]
        else:
            self.current = None
            self.drawingPolygon.emit(False)
        self.update()

    def loadPixmap(self, img, bits=None):
        self.src_image = img
        self.bits = bits
        if not self.update_image():
            return False
        self.shapes = []
        self.update()
        return True

    def update_image(self):
        if len(self.src_image) == 0:
            print(f"CanvasError: Loaded image is empty.")
            return False
        
        if self.src_image.ndim == 3:  # color image
            h, w = self.src_image.shape[0:2]
            img = self.src_image
            # img = gamma_correct(img, self.brightness)
            # img = change_contrast(img, self.contrast)
            dtype = img.dtype
            if dtype == "uint8":
                img = graylevel_transform(img, self.brightness*255, self.contrast)
                self.image = QtGui.QImage(img, w, h, img[0].nbytes, QtGui.QImage.Format_RGB888)
            elif dtype == "uint16":
                img = graylevel_transform(img, self.brightness*65535, self.contrast)
                self.image = QtGui.QImage(img, w, h, img[0].nbytes, QtGui.QImage.Format_RGB16)
            else:
                print(f"CanvasError: Unsupported {dtype}.")
                return False
        elif self.src_image.ndim == 2 and self.is_dicom:
            h, w = self.src_image.shape
            img = self.src_image
            img = dicom_transform(img, self.wc, self.ww, self.bits)
            dtype = img.dtype
            if dtype in [np.uint8, np.int8]:
                self.image = QtGui.QImage(img, w, h, img[0].nbytes, QtGui.QImage.Format_Grayscale8)
            elif dtype in [np.uint16, np.int16]:
                self.image = QtGui.QImage(img, w, h, img[0].nbytes, QtGui.QImage.Format_Grayscale16)
            else:
                print(f"CanvasError: Unsupported {dtype}.")
                return False
        elif self.src_image.ndim == 2:  # grayscale image except DICOM format
            h, w = self.src_image.shape[0:2]
            img = self.src_image
            # img = gamma_correct(img, self.brightness)
            # img = change_contrast(img, self.contrast)
            dtype = img.dtype
            if dtype == "uint8":
                img = graylevel_transform(img, self.brightness*255, self.contrast)
                self.image = QtGui.QImage(img, w, h, img[0].nbytes, QtGui.QImage.Format_Grayscale8)
            elif dtype == "uint16":
                img = graylevel_transform(img, self.brightness*65535, self.contrast)
                self.image = QtGui.QImage(img, w, h, img[0].nbytes, QtGui.QImage.Format_GrayScale16)
            else:
                print(f"CanvasError: Unsupported {dtype}.")
                return False
        else:
            print(f"CanvasError: Unsupported image of {self.src_image.ndim} dimentions.")
            return False

        self.pixmap = QtGui.QPixmap.fromImage(self.image)
        self.img_array = img
        self.update()
        return True

    def loadShapes(self, shapes, replace=True):
        if replace:
            self.shapes = list(shapes)
        else:
            self.shapes.extend(shapes)
        self.storeShapes()
        self.current = None
        self.hShape = None
        self.hVertex = None
        self.hEdge = None
        self.update()

    def setShapeVisible(self, shape, value):
        self.visible[shape] = value
        self.update()

    def overrideCursor(self, cursor):
        self._cursor = cursor
        if QtWidgets.QApplication.overrideCursor() != cursor:
            QtWidgets.QApplication.setOverrideCursor(cursor)

    def restoreCursor(self):
        self._cursor = CURSOR_DEFAULT
        if QtWidgets.QApplication.overrideCursor() != CURSOR_DEFAULT:
            QtWidgets.QApplication.restoreOverrideCursor()


    def toggle_show_label(self):
        if self.is_show_label:
            self.is_show_label = False
        else:
            self.is_show_label = True
        self.update()


    def set_target_label(self, label):
        self.target_label = label
        self.update()


    def resetState(self):
        self.restoreCursor()
        self.pixmap = None
        # self.brightness = 1.0
        # self.contrast = 0.0
        self.shapesBackups = []
        self.update()


    def reset_brightness_contrast(self):
        self.brightness = 0.0
        self.contrast = 1.0
        self.wc = self.original_wc
        self.ww = self.original_ww
        self.update_image()


    def reset_params(self):
        self.brightness = 0.0
        self.contrast = 1.0
        self.wc = 0
        self.ww = 0
        # self.update_image()