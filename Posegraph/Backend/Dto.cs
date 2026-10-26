using System.Text.Json;
using System.Text.Json.Serialization;

namespace Posegraph.Backend;

/// <summary>重定向参数，字段与后端 posegraph.tuning.RetargetParams 一致。</summary>
public sealed class RetargetParams
{
    [JsonPropertyName("amplitude")] public Dictionary<string, float> Amplitude { get; set; } = new();
    [JsonPropertyName("offsets")] public Dictionary<string, float[]> Offsets { get; set; } = new();
    [JsonPropertyName("hand_strength")] public float HandStrength { get; set; } = 1f;
    [JsonPropertyName("hand_near")] public float HandNear { get; set; } = 0.20f;
    [JsonPropertyName("hand_far")] public float HandFar { get; set; } = 0.40f;
    [JsonPropertyName("upright")] public bool Upright { get; set; } = true;
    [JsonPropertyName("upright_max_deg")] public float UprightMaxDeg { get; set; } = 30f;
    [JsonPropertyName("ground")] public bool Ground { get; set; } = true;
    [JsonPropertyName("helper_strength")] public float HelperStrength { get; set; } = 1f;

    public float Amp(string group) => Amplitude.TryGetValue(group, out var v) ? v : 1f;

    public RetargetParams Clone() => new()
    {
        Amplitude = new(Amplitude),
        Offsets = Offsets.ToDictionary(kv => kv.Key, kv => (float[])kv.Value.Clone()),
        HandStrength = HandStrength, HandNear = HandNear, HandFar = HandFar,
        Upright = Upright, UprightMaxDeg = UprightMaxDeg, Ground = Ground, HelperStrength = HelperStrength,
    };

    public static RetargetParams Preset(string name)
    {
        var p = new RetargetParams();
        if (name == "illustration")
        {
            // 插画的头颈比例夸张，识别容易把头颈角度估大（游戏实测 0.7 合适）
            p.Amplitude["neck"] = 0.7f;
            p.Amplitude["head"] = 0.7f;
        }
        return p;
    }
}

public sealed record HumanInfo(float[] Bbox);

public sealed record SessionInfo(string SessionId, string Name, int[] ImageSize, int InferenceMs, string? Device, HumanInfo[] Humans);

public sealed record Segment(float[] A, float[] B, string Side);

public sealed record Overlay(Segment[] Mhr, Segment[] Ffxiv);

public sealed record StickJoint(string Name, float[] Pos, string? Parent);

public sealed record Stick(StickJoint[] Joints, Segment[] Mhr);

public sealed record PoseWarning(string Bone, string Level, float Value, float Limit, string Message, string? Group, float? Suggest);

/// <summary>后端实际使用的骨架。Source：plugin（插件识别）/ guess（按骨长推断）/ default。</summary>
public sealed record RaceUsed(string Code, string Name, string Group, bool Validated, string Source, string[] Warnings);

public sealed record RetargetResult(
    string Pose, PoseWarning[] Warnings, Dictionary<string, string> Severity,
    JsonElement Compensation, Overlay? Overlay, Stick? Stick, float ElapsedMs, RaceUsed? Race, bool ReferenceStale = false,
    bool? Ivcs = null);   // 游戏角色是否加载了 IVCS 骨骼（读到游戏姿势时才有）

/// <summary>
/// 一张辅助视角图相对主图的情况（backend multiview.describe）。Yaw：机位绕人物转了多少度（正 = 人物右侧）；
/// View：same / left / right / back；Share：主要骨段平均采纳了多少；Conflict：主要骨段画面方向与主图平均差几度，Worst 为差得最多的骨段。
/// </summary>
public sealed record AuxView(float Yaw, string View, float Share, float Conflict, string Worst, float WorstDeg);

public sealed record ViewsResult(AuxView[] Views);

public sealed record CameraResult(float PositionErrorCm, float ForwardErrorDeg, float FovDeg, float TargetFovDeg,
                                  float FramingErrorPct, string[] Log);

public sealed record GpuInfo(string Name, int FreeMB, int TotalMB, int UsedMB);

/// <summary>Device / Precision 为当前加载的实际值，DeviceReason 为"自动"档的选择原因。</summary>
public sealed record ModelState(bool Loaded, bool Loading, string? Error, string? Device, string? Precision, string? DeviceReason);

public sealed record Health(string Status, string Version, ModelState Model, GpuInfo? Gpu, int Sessions);
