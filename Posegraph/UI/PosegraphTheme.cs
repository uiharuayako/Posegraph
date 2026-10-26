using System.Numerics;
using Dalamud.Bindings.ImGui;
using Dalamud.Interface.Utility;

namespace Posegraph.UI;

/// <summary>
/// Posegraph 自己的界面主题：深灰底、单一青蓝强调色、半透明按钮、圆角与紧凑间距（思路参考 Brio 的做法：只套在自己的窗口上，
/// 不改 Dalamud 全局主题）。窗口在 PreDraw 里 Push、PostDraw 里 Pop；尺寸按 Dalamud 的界面缩放。
/// </summary>
public static class PosegraphTheme
{
    public static readonly Vector4 Accent = Rgb(47, 181, 168);
    public static readonly Vector4 AccentHover = Rgb(64, 204, 190);
    public static readonly Vector4 AccentActive = Rgb(36, 146, 136);

    // 状态色（文字 / 图标）：在深灰底上都清楚
    public static readonly Vector4 Green = Rgb(96, 214, 120);
    public static readonly Vector4 Yellow = Rgb(240, 196, 80);
    public static readonly Vector4 Red = Rgb(240, 98, 88);
    public static readonly Vector4 Grey = Rgb(150, 154, 160);
    public static readonly Vector4 Blue = Rgb(110, 175, 250);

    private static Vector4 Rgb(int r, int g, int b, float a = 1f) => new(r / 255f, g / 255f, b / 255f, a);
    private static Vector4 Alpha(Vector4 c, float a) => c with { W = a };

    private static readonly (ImGuiCol Col, Vector4 Value)[] Colors =
    [
        (ImGuiCol.Text, Rgb(234, 236, 240)),
        (ImGuiCol.TextDisabled, Rgb(132, 136, 143)),
        (ImGuiCol.WindowBg, Rgb(23, 24, 27, 0.97f)),
        (ImGuiCol.ChildBg, Rgb(0, 0, 0, 0)),
        (ImGuiCol.PopupBg, Rgb(28, 29, 33, 0.98f)),
        (ImGuiCol.Border, Rgb(52, 54, 60)),
        (ImGuiCol.BorderShadow, Rgb(0, 0, 0, 0)),
        (ImGuiCol.FrameBg, Rgb(40, 42, 47)),
        (ImGuiCol.FrameBgHovered, Rgb(51, 54, 60)),
        (ImGuiCol.FrameBgActive, Rgb(58, 62, 69)),
        (ImGuiCol.TitleBg, Rgb(23, 24, 27)),
        (ImGuiCol.TitleBgActive, Rgb(30, 32, 36)),
        (ImGuiCol.TitleBgCollapsed, Rgb(23, 24, 27, 0.9f)),
        (ImGuiCol.MenuBarBg, Rgb(32, 34, 38)),
        (ImGuiCol.ScrollbarBg, Rgb(0, 0, 0, 0)),
        (ImGuiCol.ScrollbarGrab, Rgb(62, 64, 70)),
        (ImGuiCol.ScrollbarGrabHovered, Rgb(80, 83, 90)),
        (ImGuiCol.ScrollbarGrabActive, Accent),
        (ImGuiCol.CheckMark, Accent),
        (ImGuiCol.SliderGrab, Alpha(Accent, 0.85f)),
        (ImGuiCol.SliderGrabActive, AccentHover),
        (ImGuiCol.Button, Rgb(255, 255, 255, 0.08f)),
        (ImGuiCol.ButtonHovered, Alpha(Accent, 0.45f)),
        (ImGuiCol.ButtonActive, Alpha(Accent, 0.70f)),
        (ImGuiCol.Header, Rgb(255, 255, 255, 0.06f)),
        (ImGuiCol.HeaderHovered, Alpha(Accent, 0.28f)),
        (ImGuiCol.HeaderActive, Alpha(Accent, 0.45f)),
        (ImGuiCol.Separator, Rgb(58, 60, 66)),
        (ImGuiCol.SeparatorHovered, Alpha(Accent, 0.6f)),
        (ImGuiCol.SeparatorActive, Accent),
        (ImGuiCol.ResizeGrip, Rgb(0, 0, 0, 0)),
        (ImGuiCol.ResizeGripHovered, Alpha(Accent, 0.4f)),
        (ImGuiCol.ResizeGripActive, Accent),
        (ImGuiCol.Tab, Rgb(34, 36, 40)),
        (ImGuiCol.TabHovered, Alpha(Accent, 0.55f)),
        (ImGuiCol.TabActive, AccentActive),
        (ImGuiCol.TabUnfocused, Rgb(30, 31, 35)),
        (ImGuiCol.TabUnfocusedActive, Rgb(40, 104, 98)),
        (ImGuiCol.TableHeaderBg, Rgb(36, 38, 42)),
        (ImGuiCol.TableBorderStrong, Rgb(58, 60, 66)),
        (ImGuiCol.TableBorderLight, Rgb(44, 46, 51)),
        (ImGuiCol.TableRowBg, Rgb(0, 0, 0, 0)),
        (ImGuiCol.TableRowBgAlt, Rgb(255, 255, 255, 0.03f)),
        (ImGuiCol.TextSelectedBg, Alpha(Accent, 0.35f)),
        (ImGuiCol.DragDropTarget, Accent),
        (ImGuiCol.NavHighlight, Alpha(Accent, 0.7f)),
    ];

    private static readonly (ImGuiStyleVar Var, Vector2 Value)[] Vectors =
    [
        (ImGuiStyleVar.WindowPadding, new(8, 8)),
        (ImGuiStyleVar.FramePadding, new(6, 4)),
        (ImGuiStyleVar.CellPadding, new(4, 3)),
        (ImGuiStyleVar.ItemSpacing, new(6, 5)),
        (ImGuiStyleVar.ItemInnerSpacing, new(5, 4)),
    ];

    private static readonly (ImGuiStyleVar Var, float Value)[] Floats =
    [
        (ImGuiStyleVar.IndentSpacing, 18),
        (ImGuiStyleVar.ScrollbarSize, 10),
        (ImGuiStyleVar.GrabMinSize, 12),
        (ImGuiStyleVar.WindowBorderSize, 1),
        (ImGuiStyleVar.ChildBorderSize, 1),
        (ImGuiStyleVar.PopupBorderSize, 1),
        (ImGuiStyleVar.FrameBorderSize, 0),
        (ImGuiStyleVar.WindowRounding, 8),
        (ImGuiStyleVar.ChildRounding, 5),
        (ImGuiStyleVar.FrameRounding, 5),
        (ImGuiStyleVar.PopupRounding, 6),
        (ImGuiStyleVar.ScrollbarRounding, 6),
        (ImGuiStyleVar.GrabRounding, 4),
        (ImGuiStyleVar.TabRounding, 5),
    ];

    // 边框粗细不随缩放变
    private static readonly HashSet<ImGuiStyleVar> Unscaled =
        [ImGuiStyleVar.WindowBorderSize, ImGuiStyleVar.ChildBorderSize, ImGuiStyleVar.PopupBorderSize, ImGuiStyleVar.FrameBorderSize];

    public static void Push()
    {
        var scale = ImGuiHelpers.GlobalScale;
        foreach (var (col, value) in Colors) ImGui.PushStyleColor(col, value);
        foreach (var (v, value) in Vectors) ImGui.PushStyleVar(v, value * scale);
        foreach (var (v, value) in Floats) ImGui.PushStyleVar(v, Unscaled.Contains(v) ? value : value * scale);
    }

    public static void Pop()
    {
        ImGui.PopStyleVar(Vectors.Length + Floats.Length);
        ImGui.PopStyleColor(Colors.Length);
    }
}
