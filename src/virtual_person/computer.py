from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class UIElement:
    element_id: str
    role: str
    text: str = ""
    value: str = ""
    focused: bool = False
    enabled: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.element_id,
            "role": self.role,
            "text": self.text,
            "value": self.value,
            "focused": self.focused,
            "enabled": self.enabled,
        }


@dataclass(slots=True)
class VirtualComputer:
    """A deterministic, host-isolated computer simulation."""

    powered_on: bool = False
    active_app: str = "desktop"
    focused_element: str | None = None
    notes_text: str = ""
    browser_address: str = ""
    browser_page: str = "Home"
    terminal_lines: list[str] = field(default_factory=list)
    mouse_x: int = 0
    mouse_y: int = 0
    task_complete: bool = False

    def power_on(self) -> str:
        self.powered_on = True
        self.active_app = "desktop"
        self.focused_element = None
        return "The virtual computer powered on."

    def launch(self, app: str) -> str:
        if not self.powered_on:
            self.power_on()
        app = app.lower()
        if app not in {"desktop", "notes", "browser", "terminal"}:
            raise ValueError(f"Unknown virtual application: {app}")
        self.active_app = app
        self.focused_element = {
            "notes": "notes_editor",
            "browser": "address_bar",
            "terminal": "terminal_input",
        }.get(app)
        return f"Opened {app}."

    def click(self, target: str | None = None) -> str:
        if not self.powered_on:
            raise RuntimeError("The virtual computer is off.")
        if target in {"notes_icon", "notes_editor"}:
            self.launch("notes")
        elif target in {"browser_icon", "address_bar"}:
            self.launch("browser")
        elif target in {"terminal_icon", "terminal_input"}:
            self.launch("terminal")
        elif target == "submit_task":
            self.task_complete = bool(self.notes_text.strip())
            return "Submitted the current notes." if self.task_complete else "Nothing to submit."
        elif target is not None:
            self.focused_element = target
        return f"Clicked {target or 'the screen'}."

    def type_text(self, text: str) -> str:
        if not self.powered_on:
            raise RuntimeError("The virtual computer is off.")
        if self.active_app == "notes":
            self.notes_text += text
            return f"Typed {len(text)} characters into Notes."
        if self.active_app == "browser":
            self.browser_address += text
            return f"Typed into the address bar: {text!r}."
        if self.active_app == "terminal":
            if not self.terminal_lines:
                self.terminal_lines.append("")
            self.terminal_lines[-1] += text
            return f"Typed into the virtual terminal: {text!r}."
        raise RuntimeError("No editable application is open.")

    def press_key(self, key: str) -> str:
        if not self.powered_on:
            raise RuntimeError("The virtual computer is off.")
        key = key.lower()
        if key == "enter" and self.active_app == "browser":
            address = self.browser_address.strip()
            self.browser_page = self._lookup_page(address)
            return f"Loaded virtual page {self.browser_page!r}."
        if key == "enter" and self.active_app == "terminal":
            command = self.terminal_lines[-1] if self.terminal_lines else ""
            output = self._run_virtual_command(command)
            self.terminal_lines.extend([output, ""])
            return output
        if key == "backspace":
            if self.active_app == "notes":
                self.notes_text = self.notes_text[:-1]
            elif self.active_app == "browser":
                self.browser_address = self.browser_address[:-1]
            elif self.active_app == "terminal" and self.terminal_lines:
                self.terminal_lines[-1] = self.terminal_lines[-1][:-1]
            return "Pressed Backspace."
        return f"Pressed {key}."

    def move_mouse(self, x: int, y: int) -> str:
        self.mouse_x = max(0, min(1919, int(x)))
        self.mouse_y = max(0, min(1079, int(y)))
        return f"Moved the virtual mouse to ({self.mouse_x}, {self.mouse_y})."

    def observe(self) -> dict[str, Any]:
        return {
            "powered_on": self.powered_on,
            "active_app": self.active_app,
            "mouse": [self.mouse_x, self.mouse_y],
            "elements": [element.as_dict() for element in self._elements()],
            "task_complete": self.task_complete,
        }

    def _elements(self) -> list[UIElement]:
        if not self.powered_on:
            return [UIElement("power_button", "button", text="Power")]
        if self.active_app == "desktop":
            return [
                UIElement("notes_icon", "button", text="Notes"),
                UIElement("browser_icon", "button", text="Browser"),
                UIElement("terminal_icon", "button", text="Terminal"),
            ]
        if self.active_app == "notes":
            return [
                UIElement(
                    "notes_editor",
                    "textbox",
                    value=self.notes_text,
                    focused=self.focused_element == "notes_editor",
                ),
                UIElement("submit_task", "button", text="Submit"),
            ]
        if self.active_app == "browser":
            return [
                UIElement(
                    "address_bar",
                    "textbox",
                    value=self.browser_address,
                    focused=self.focused_element == "address_bar",
                ),
                UIElement("page_title", "heading", text=self.browser_page),
            ]
        return [
            UIElement(
                "terminal_output",
                "document",
                text="\n".join(self.terminal_lines[-20:]),
            ),
            UIElement(
                "terminal_input",
                "textbox",
                value=self.terminal_lines[-1] if self.terminal_lines else "",
                focused=True,
            ),
        ]

    @staticmethod
    def _lookup_page(address: str) -> str:
        pages = {
            "home": "Home",
            "kitchen.local": "Kitchen Reference",
            "schedule.local": "Daily Schedule",
            "help.local": "Computer Help",
        }
        return pages.get(address.lower(), "Page Not Found")

    @staticmethod
    def _run_virtual_command(command: str) -> str:
        command = command.strip()
        if command == "help":
            return "Commands: help, date, echo <text>, clear"
        if command == "date":
            return "Virtual system time is controlled by the apartment simulation."
        if command.startswith("echo "):
            return command[5:]
        if command == "clear":
            return ""
        return f"Command not found: {command}"
