from core.tk.component.component import Component
from core.tk.event import KeyEvent
from core.tk.rendering import UIRenderingContext, RenderingResult


class SeparatorComponent(Component):
    def render(self, ctx: UIRenderingContext) -> RenderingResult:
        side_margin = 4
        top_margin = 5
        ctx.canvas.line(
            start=(ctx.left + side_margin, ctx.top + top_margin),
            end=(ctx.left + ctx.max_width - side_margin, ctx.top + top_margin),
            color=ctx.fg_color,
            edge_color=ctx.edge_color,
        )
        return RenderingResult(
            height=top_margin * 2 + 1,
        )

    def key_event(self, event: KeyEvent) -> bool:
        return super().key_event(event)

    def focus_count(self) -> int:
        return 0
