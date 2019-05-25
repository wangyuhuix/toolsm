import tools
import os.path as osp
import json
import os
import itertools
import numpy as np



def flatten(unflattened, parent_key='', separator='.'):
    items = []
    for k, v in unflattened.items():
        if separator in k:
            raise ValueError(
                "Found separator ({}) from key ({})".format(separator, k))
        new_key = parent_key + separator + k if parent_key else k
        if isinstance(v, collections.MutableMapping) and v:
            items.extend(flatten(v, new_key, separator=separator).items())
        else:
            items.append((new_key, v))

    return dict(items)

def unflatten(flattened, separator='.'):
    result = {}
    for key, value in flattened.items():
        parts = key.split(separator)
        d = result
        for part in parts[:-1]:
            if part not in d:
                d[part] = {}
            d = d[part]
        d[parts[-1]] = value

    return result


def int2str(i):
    if isinstance(i, int) and i >= int(1e4):
        i = f'{i:.0e}'
    return i




def prepare_dirs(args, args_dict=None, key_first=None, keys_exclude=[], dirs_type=['log'], name_project='tmpProject'):
    '''
    required keys in args: 'force_write','name_group','keys_group'

    parser.add_argument('--force_write', default=1, type=int)
    parser.add_argument('--name_group', default='tmp', type=str)
    parser.add_argument('--keys_group', default=['clipped_type'], type=ast.literal_eval)

    root_dir/name_project/dir_type/name_group/name_task
    root_dir: root dir
    name_project: your project
    dir_type: e.g. log, model
    name_group: for different setting, e.g. hyperparameter or just for test
    '''
    SPLIT = ','


    if args_dict is None:
        args_dict = vars(args)
    force_write = args.force_write


    name_key_group = ''
    for i,key in enumerate(args.keys_group):
        if i>0:
            name_key_group += SPLIT
        name_key_group += f'{key}={int2str(args_dict[key])}'

    args.name_group = name_key_group + (SPLIT if name_key_group else '') + args.name_group

    if not args.name_group:
        args.name_group = 'tmpGroup'
        print( f'args.name_group is empty. it is set to be {args.name_task}' )

    # -------------- get root directory -----------
    if tools.ispc('xiaoming'):
        root_dir = '/media/d/e/et'
    else:
        root_dir = f"{os.environ['HOME']}/xm/et"

    root_dir = f'{root_dir}/{name_project}'

    # ----------- get sub directory -----------
    keys_exclude.extend(['force_write','name_group','keys_group'])
    if key_first is not None:
        keys_exclude.append(key_first)
    keys_exclude.extend( args.keys_group )
    # --- add first key
    if key_first is None:
        key_first = list(args_dict.keys())[0]
    keys_exclude.append(key_first)
    name_task = args_dict[key_first]

    # --- add keys common
    for key in args_dict.keys():
        if key not in keys_exclude:
            name_task += f'{SPLIT}{key}={int2str(args_dict[key])}'
    # name_task += ('' if name_suffix == '' else f'{split}{name_suffix}')

    # ----------------- prepare directory ----------
    def get_dir_full( d_type, suffix='' ):
        return f'{root_dir}/{d_type}/{args.name_group}/{name_task}{suffix}'

    dirs_full = dict()
    for d_type in dirs_type:
        assert d_type
        dirs_full[d_type] = get_dir_full(d_type)
        # print(dirs_full[d_type])
        setattr( args, f'{d_type}_dir', dirs_full[d_type] )
    # exit()
    # ----- Move Dirs
    if np.any( [osp.exists( d ) for d in dirs_full.values() ]):  # 如果文件夹存在，则删除
        # print(
        #     f"Exsits sub directory: {name_task} in {root_dir} \nMove to discard(y or n)?",
        #     end='')
        # if force_write > 0:
        #     cmd = 'y'
        #     print(f'y (auto by force_write={force_write})')
        # elif force_write < 0:
        #     exit()
        # else:
        #     cmd = input()
        cmd = 'y'
        if cmd == 'y':
            for i in itertools.count():
                if i == 0:
                    suffix = ''
                else:
                    suffix = f'{SPLIT}{i}'
                dirs_full_discard = {}
                for d_type in dirs_type:

                    dirs_full_discard[d_type] = get_dir_full( f'{d_type}_del', suffix )
                if not np.any( [osp.exists( d ) for d in dirs_full_discard.values() ]):
                    break

            print(tools.colorize(f"Going to move \n{name_task} \n{name_task}{suffix}\n"+f"Confirm move(y or n)?", 'red'), end='')
            if force_write > 0:
                cmd = 'y'
                print(f'y (auto by force_write={force_write})')
            elif force_write < 0:
                exit()
            else:
                cmd = input()

            if cmd == 'y' \
                and \
                np.all(tools.check_safe_path( dirs_full[d_mid], confirm=False) for d_type in dirs_type):
                import shutil
                for d_type in dirs_type:
                    if osp.exists(dirs_full[d_type]):
                        print( tools.colorize( f'Move:\n{dirs_full[d_type]}\n To\n {dirs_full_discard[d_type]}','red') )
                        shutil.move(dirs_full[d_type], dirs_full_discard[d_type])#TODO: test if not exist?
            else:
                print("Please Rename 'name_group'")
                exit()
        else:
            print("Please Rename 'name_group'")
            exit()


    for d_type in dirs_type:
        tools.makedirs( dirs_full[d_type] )

    with open(f'{args.log_dir}/args.json', 'w') as f:
        json.dump(vars(args), f, indent=4, separators=(',', ':'))




if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument('--env', help='environment ID', type=str, default='Reacher-v2')
    parser.add_argument('--env', help='environment ID', type=str, default='Swimmer-v2')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num-timesteps', type=int, default=int(1e6))
    parser.add_argument('--play', default=False, action='store_true')
    parser.add_argument('--clipped-type', default='origin', type=str)

    parser.add_argument('--name_group', default='', type=str)
    parser.add_argument('--force-write', default=1, type=int)

    args = parser.parse_args()
    args_dict = vars(args)
    args_dict['num_timesteps'] = f"{args_dict['num_timesteps']:.0e}"
    prepare_dirs( args, args_dict )
    exit()