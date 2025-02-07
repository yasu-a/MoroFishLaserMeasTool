from app_tk.component.component import Component
from app_tk.event import KeyEvent
from app_tk.rendering import RenderingContext, RenderingResult


class SpacerComponent(Component):
    def render(self, ctx: RenderingContext) -> RenderingResult:
        return RenderingResult(
            height=8,
        )

    def key_event(self, event: KeyEvent) -> bool:
        return super().key_event(event)

    def focus_count(self) -> int:
        return 0
