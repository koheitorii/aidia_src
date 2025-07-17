from qtpy import QtCore
from qtpy import QtGui
from qtpy import QtWidgets

from aidia import qt

class ToolBar(QtWidgets.QToolBar):
    """A custom toolbar that allows adding buttons with actions."""

    def __init__(self, title):
        super(ToolBar, self).__init__(title)
        layout = self.layout()
        m = (0, 0, 0, 0)
        layout.setSpacing(0)
        layout.setContentsMargins(*m)
        self.setContentsMargins(*m)
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.FramelessWindowHint)

        self.separator_num = 0  # セパレーター番号の初期化
        self._actions = {}

    def addAction(self, action: QtWidgets.QAction):
        """Add an action to the toolbar with a custom button."""
        if isinstance(action, QtWidgets.QWidgetAction):
            return super(ToolBar, self).addAction(action)
        btn = ToolButton()
        btn.setDefaultAction(action)
        btn.setToolButtonStyle(self.toolButtonStyle())
        btn.setMinimumWidth(100)
        btn.setMaximumWidth(200)
        a = self.addWidget(btn)
        self._actions[action.text()] = [a, True]
    
    def addSeparator(self):
        separator = super().addSeparator()
        self._actions['separator' + str(self.separator_num)] = [None, True]
        self.separator_num += 1
        return separator

    def updateShowButtons(self):
        """Update the visibility of buttons based on their actions."""
        self.clear()
        self.separator_num = 0  # セパレーター番号をリセット
        for i in range(len(self._actions.keys())):
            k = list(self._actions.keys())[i]
            a, is_show = self._actions[k]
            if a is None:
                self.addSeparator()
            elif is_show:
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
