using System.Reflection;
using System.Runtime.Loader;
using UiPreview;

// Dalamud 与插件从原位置加载（不复制到输出目录）。用到 Dalamud 类型的代码都在 Cli 里，
// 这里先挂上解析器（Main 被编译时不能引用它们，否则在解析器注册前就要加载）。
var dalamudDir = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData), @"XIVLauncherCN\addon\Hooks\dev");
var pluginDir = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, @"..\..\..\..\..\Posegraph\bin\Release"));
AssemblyLoadContext.Default.Resolving += (_, name) =>
{
    foreach (var dir in new[] { pluginDir, dalamudDir })
    {
        var path = Path.Combine(dir, name.Name + ".dll");
        if (File.Exists(path)) return Assembly.LoadFrom(path);
    }
    return null;
};
return Cli.Run(args, dalamudDir);
