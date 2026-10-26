using System.Numerics;
using Dalamud.Bindings.ImGui;
using Dalamud.Interface.Textures;
using Dalamud.Interface.Textures.TextureWraps;
using Dalamud.Plugin;
using Dalamud.Plugin.Services;
using SkiaSharp;

namespace UiPreview;

/// <summary>Skia 图片当作 Dalamud 纹理（Handle 是渲染器里的编号）。</summary>
public sealed class SkTexture(SkiaRenderer renderer, SKImage image) : IDalamudTextureWrap
{
    private readonly ulong _id = renderer.Register(image);

    public ImTextureID Handle => new(_id);
    public int Width => image.Width;
    public int Height => image.Height;
    public Vector2 Size => new(Width, Height);
    public IDalamudTextureWrap CreateWrapSharingLowLevelResource() => this;
    public void Dispose() => renderer.Unregister(_id);
}

/// <summary>插件用到的 Dalamud 服务的离线替身。</summary>
public static class Services
{
    public static string ConfigDir { get; } = Path.Combine(Path.GetTempPath(), "PosegraphUiPreview");

    public static IDalamudPluginInterface PluginInterface(Func<IEnumerable<IExposedPlugin>> plugins)
        => Stub.Create<IDalamudPluginInterface>(s => s
            .On("GetPluginConfigDirectory", _ => Directory.CreateDirectory(ConfigDir).FullName)
            .On("get_InstalledPlugins", _ => plugins().ToList())
            .Returns("get_UiLanguage", "zh"));

    public static IExposedPlugin Plugin(string name, string version, bool loaded = true)
        => Stub.Create<IExposedPlugin>(s => s
            .Returns("get_InternalName", name)
            .Returns("get_Name", name)
            .Returns("get_IsLoaded", loaded)
            .Returns("get_Version", Version.Parse(version)));

    public static ITextureProvider Textures(SkiaRenderer renderer)
        => Stub.Create<ITextureProvider>(s => s
            .On("CreateFromImageAsync", a =>
            {
                var bytes = a[0] is ReadOnlyMemory<byte> m ? m.ToArray() : ReadAll((Stream)a[0]!);
                var image = SKImage.FromEncodedData(bytes) ?? throw new InvalidDataException("无法解码图片");
                return Task.FromResult<IDalamudTextureWrap>(new SkTexture(renderer, image));
            })
            .On("CreateFromExistingTextureAsync", a =>
            {
                var src = (IDalamudTextureWrap)a[0]!;
                var args = (TextureModificationArgs)a[1]!;
                var image = renderer.Get(src.Handle.Handle)!;
                var w = args.NewWidth > 0 ? args.NewWidth : image.Width;
                var h = args.NewHeight > 0 ? args.NewHeight : image.Height;
                using var surface = SKSurface.Create(new SKImageInfo(w, h));
                surface.Canvas.DrawImage(image, new SKRect(0, 0, w, h), new SKSamplingOptions(SKFilterMode.Linear, SKMipmapMode.Linear));
                return Task.FromResult<IDalamudTextureWrap>(new SkTexture(renderer, surface.Snapshot()));
            })
            .Returns("HasClipboardImage", false));

    private static byte[] ReadAll(Stream s)
    {
        using var ms = new MemoryStream();
        s.CopyTo(ms);
        return ms.ToArray();
    }

    /// <summary>RunOnFrameworkThread 直接在当前线程执行；DelayTicks 立即完成。</summary>
    public static IFramework Framework()
        => Stub.Create<IFramework>(s => s
            .On("RunOnFrameworkThread", (m, a) =>
            {
                var result = ((Delegate)a[0]!).DynamicInvoke();
                if (result is Task) return result;
                var type = m.ReturnType.IsGenericType ? m.ReturnType.GetGenericArguments()[0] : null;
                return type is null ? Task.CompletedTask
                    : typeof(Task).GetMethod(nameof(Task.FromResult))!.MakeGenericMethod(type).Invoke(null, [result]);
            })
            .On("DelayTicks", _ => Task.CompletedTask));
}
