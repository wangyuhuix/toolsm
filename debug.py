
import os
from . import tools
def equal_with_previous_run(objs, iteration, name, path):
    file = os.path.join(path, f'{name}_{iteration}.pkl')
    if not os.path.exists(file):
        tools.save_vars(file, objs, disp=True)
    else:
        print(f"{'':->10}iteration {iteration}, obj: {name}{'':->10}")
        objs_old = tools.load_vars(file)
        equal(objs, objs_old)

def equal( objs, objs_old ):
    if isinstance(objs, dict):
        assert list(objs.keys()) == list(objs_old.keys())
        for i, (d1, d2) in enumerate(zip(objs.values(), objs_old.values())):
            is_equal = tools.equal(d1, d2)
            color = 'blue' if is_equal else 'red'
            print(f" key {list(objs.keys())[i]}: {tools.colorize(str(is_equal), color)}")
    else:
        for i, (d1, d2) in enumerate(zip(objs, objs_old)):
            is_equal = tools.equal(d1, d2)
            color = 'blue' if is_equal else 'red'
            print(f"item {i}: {tools.colorize(str(is_equal), color)}")
