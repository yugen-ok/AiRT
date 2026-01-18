
import sys
import inspect
import os

def print_source(source, out=sys.stdout):
    print('Name:', source['name'], file=out)
    print('Class:', source['class'], file=out)
    print('File:', source['file'], file=out)
    print('-------------------------', file=out)
    print('Source:\n\n', source['source'], file=out)
    print('=========================', file=out)
    print(file=out)

def capture_runtime_sources(func, *args, **kwargs):
    seen = set()
    collected = []

    def resolve_class(frame):
        locals_ = frame.f_locals

        if "self" in locals_:
            return type(locals_["self"])
        if "cls" in locals_ and inspect.isclass(locals_["cls"]):
            return locals_["cls"]

        return None  # plain function / module-level

    def tracer(frame, event, arg):
        if event == "call":
            code = frame.f_code
            if code in seen:
                return tracer

            seen.add(code)

            cls = resolve_class(frame)

            try:
                file = inspect.getsourcefile(code)
                source = inspect.getsource(code)
            except (OSError, IOError):
                file = code.co_filename
                source = "<source not available>"

            if cls:
                qualname = f"{cls.__module__}.{cls.__qualname__}.{code.co_name}"
            else:
                qualname = f"{code.co_filename}:{code.co_name}"

            collected.append({
                "name": qualname,
                "file": os.path.abspath(file),
                "class": None if not cls else f"{cls.__module__}.{cls.__qualname__}",
                "source": source
            })

        return tracer

    sys.settrace(tracer)
    try:
        func(*args, **kwargs)
    finally:
        sys.settrace(None)

    return collected



if __name__ == "__main__":
    def a():
        b()


    def b():
        c()


    def c():
        print("hello")


    sources = capture_runtime_sources(a)

    for name, info in sources.items():
        print("\n" + "=" * 60)
        print(f"FUNCTION: {name}")
        print(f"FILE: {info['file']}")
        print(info["source"])

