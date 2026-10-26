using System.Numerics;
using Dalamud.Bindings.ImGui;
using Dalamud.Interface.Utility.Raii;

namespace Posegraph.UI;

/// <summary>单行文字放不下时的处理：截成「…」结尾，悬停显示全文（窗口窄、英文更长时不会被生硬裁掉）。</summary>
public static class UiText
{
    /// <summary>截到 width 宽以内，放不下时以「…」结尾。</summary>
    public static string Ellipsize(string text, float width)
    {
        if (width <= 0) return "";
        if (ImGui.CalcTextSize(text).X <= width) return text;
        // 二分找能放下的最长前缀
        int lo = 0, hi = text.Length;
        while (lo < hi)
        {
            var mid = (lo + hi + 1) / 2;
            if (ImGui.CalcTextSize(text[..mid] + "…").X <= width) lo = mid;
            else hi = mid - 1;
        }
        return lo == 0 ? "…" : text[..lo] + "…";
    }

    /// <summary>彩色单行文字，按剩余宽度截断，截断时悬停显示全文。</summary>
    public static void Fit(string text, Vector4 color, float? width = null)
    {
        var shown = Ellipsize(text, width ?? ImGui.GetContentRegionAvail().X);
        using (ImRaii.PushColor(ImGuiCol.Text, color))
            ImGui.TextUnformatted(shown);
        if (shown != text && ImGui.IsItemHovered()) ImGui.SetTooltip(text);
    }

    /// <summary>下拉框的预览文字：去掉右侧箭头占的宽度后截断。</summary>
    public static string ComboPreview(string text, float comboWidth)
        => Ellipsize(text, comboWidth - ImGui.GetFrameHeight() - ImGui.GetStyle().FramePadding.X * 2);
}
