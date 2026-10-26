using Dalamud.Bindings.ImGui;
using Dalamud.Interface.Utility;
using Posegraph.Backend;

namespace Posegraph.UI;

public sealed partial class MainWindow
{
    private bool DrawRecognitionConcurrencySettings()
    {
        var changed = false;
        var parallelism = Math.Clamp(_config.BatchRecognitionParallelism, 1, 8);
        ImGui.TextUnformatted(Loc.T("多图识别并行数"));
        ImGui.SetNextItemWidth(70 * ImGuiHelpers.GlobalScale);
        if (ImGui.BeginCombo("##imageParallelism", parallelism.ToString()))
        {
            for (var count = 1; count <= 8; count++)
                if (ImGui.Selectable(count.ToString(), parallelism == count))
                {
                    parallelism = count;
                    _config.BatchRecognitionParallelism = count;
                    changed = true;
                }
            ImGui.EndCombo();
        }
        ItemTip(Loc.T("默认 1，按顺序识别；可设为 1～8，下批生效。默认多进程策略会加载多份模型；显式开启共享模式后只保留一份权重，以下预算不适用。"));

        // Default-process planning allowances; shared mode does not use this budget.
        // These describe the selected setting, not the active batch or
        // live process readings. Auto retains the GPU scenario with a fallback note.
        var estimate = RecognitionResources.Estimate(_config.BackendDevice, parallelism, _config.MhrOnCpu);
        var vram = estimate.VramMB / 1024f;
        var ram = estimate.RamMB / 1024f;
        var vramLow = estimate.VramMB > 0 && _probe.VramFreeMB is { } free && estimate.VramMB > free;
        var ramLow = _probe.RamTotalMB > 0 && estimate.RamMB > _probe.RamAvailMB;
        var color = vramLow || ramLow ? Yellow : Grey;
        var summary = Loc.F("多进程预算：显存约 {0:0.0} GB / 内存约 {1:0.0} GB", vram, ram);
        ImGui.SameLine();
        ImGui.BeginGroup();
        if (ImGui.CalcTextSize(summary).X <= ImGui.GetContentRegionAvail().X)
            ImGui.TextColored(color, summary);
        else
        {
            // Keep both estimates to the right of the combo in narrow windows.
            ImGui.TextColored(color, Loc.F("多进程显存预算约 {0:0.0} GB", vram));
            ImGui.TextColored(color, Loc.F("多进程内存预算约 {0:0.0} GB", ram));
        }
        ImGui.EndGroup();
        if (ImGui.IsItemHovered())
        {
            var tip = Loc.F("按默认多进程策略的 {0} 份模型估算预算，包含冷加载余量；不是实时占用或保证峰值，共享模式不适用。", estimate.Workers)
                + "\n" + Loc.T("不含游戏、图片与会话缓存，以及已有 CUDA 上下文残留。");
            if (estimate.Scenario == RecognitionResourceScenario.Auto)
                tip += "\n" + Loc.T("自动模式按 GPU 方案估算；显存不足时可能改用 CPU，实际占用会变化。");
            else if (estimate.Scenario == RecognitionResourceScenario.Cpu)
                tip += "\n" + Loc.T("CPU 模式不为识别分配显存；先前 GPU 进程的残留需等进程退出后释放。");
            var availableVram = _probe.VramFreeMB is { } available ? Loc.F("{0:0.0} GB", available / 1024f) : Loc.T("未知");
            var availableRam = _probe.RamTotalMB > 0 ? Loc.F("{0:0.0} GB", _probe.RamAvailMB / 1024f) : Loc.T("未知");
            tip += "\n" + Loc.F("当前可用：显存 {0} / 内存 {1}", availableVram, availableRam);
            if (vramLow || ramLow)
                tip += "\n" + Loc.T("默认多进程预算超过当前可用量，可降低并行数；共享模式不适用此预算。");
            ImGui.SetTooltip(tip);
        }
        return changed;
    }
}
