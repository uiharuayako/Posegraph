using System.Reflection;

namespace UiPreview;

/// <summary>
/// 任意接口的替身（Dalamud 服务）：没指定的方法返回默认值（Task 返回已完成的、集合返回空），
/// 指定的按方法名处理（属性用 get_Xxx）。
/// </summary>
public class Stub : DispatchProxy
{
    public Dictionary<string, Func<MethodInfo, object?[], object?>> Handlers { get; } = new();

    public static T Create<T>(Action<Stub>? setup = null) where T : class
    {
        var proxy = Create<T, Stub>();
        setup?.Invoke((Stub)(object)proxy);
        return proxy;
    }

    public Stub On(string name, Func<object?[], object?> handler)
    {
        Handlers[name] = (_, a) => handler(a);
        return this;
    }

    public Stub On(string name, Func<MethodInfo, object?[], object?> handler)
    {
        Handlers[name] = handler;
        return this;
    }

    public Stub Returns(string name, object? value) => On(name, _ => value);

    protected override object? Invoke(MethodInfo? method, object?[]? args)
    {
        if (method is null) return null;
        if (Handlers.TryGetValue(method.Name, out var handler)) return handler(method, args ?? []);
        return Default(method.ReturnType);
    }

    public static object? Default(Type type)
    {
        if (type == typeof(void)) return null;
        if (type == typeof(string)) return "";
        if (type == typeof(Task)) return Task.CompletedTask;
        if (type.IsGenericType && type.GetGenericTypeDefinition() == typeof(Task<>))
        {
            var inner = Default(type.GetGenericArguments()[0]);
            return typeof(Task).GetMethod(nameof(Task.FromResult))!.MakeGenericMethod(type.GetGenericArguments()[0]).Invoke(null, [inner]);
        }
        if (type.IsArray) return Array.CreateInstance(type.GetElementType()!, 0);
        if (type.IsGenericType && type.GetGenericArguments().Length == 1)
        {
            var list = typeof(List<>).MakeGenericType(type.GetGenericArguments()[0]);
            if (type.IsAssignableFrom(list)) return Activator.CreateInstance(list);
        }
        return type.IsValueType ? Activator.CreateInstance(type) : null;
    }
}
