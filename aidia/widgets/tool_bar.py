from qtpy import QtCore
from qtpy import QtWidgets


class ToolBar(QtWidgets.QToolBar):

    def __init__(self, title):
        super(ToolBar, self).__init__(title)
        layout = self.layout()
        m = (0, 0, 0, 0)
        layout.setSpacing(0)
        layout.setContentsMargins(*m)
        self.setContentsMargins(*m)
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.FramelessWindowHint)
        
        self._actions = []
        self.flags = []


    def addAction(self, action):
        if isinstance(action, QtWidgets.QWidgetAction):
            return super(ToolBar, self).addAction(action)
        btn = ToolButton()
        btn.setDefaultAction(action)
        btn.setToolButtonStyle(self.toolButtonStyle())
        btn.setMinimumWidth(100)
        btn.setMaximumWidth(200)
        # btn.setMinimumHeight(80)
        # btn.setMaximumHeight(80)
        a = self.addWidget(btn)
        self._actions.append(a)
        self.flags.append(True)


    def updateShowButtons(self):
        self.clear()
        for a, flag in zip(self._actions, self.flags):
            if flag:
                self.addAction(a)


class ToolButton(QtWidgets.QToolButton):

    """ToolBar companion class which ensures all buttons have the same size."""

    minSize = (60, 60)

    def minimumSizeHint(self):
        ms = super(ToolButton, self).minimumSizeHint()
        w1, h1 = ms.width(), ms.height()
        w2, h2 = self.minSize
        self.minSize = max(w1, w2), max(h1, h2)
        return QtCore.QSize(*self.minSize)
