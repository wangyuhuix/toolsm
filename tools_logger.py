import tools
import os.path as osp
import json
import os
import itertools
import numpy as np
def prepare_dirs(args, args_dict=None, key_first=None, keys_exclude=[], name_suffix='', dirs_mid=['log'], name_project='tmp'):
    if args_dict is None:
        args_dict = vars(args)
    force_write = args.force_write

    # -------------- get root directory -----------
    if tools.ispc('xiaoming'):
        root_dir = '/media/d/e/et'
    else:
        root_dir = f"{os.environ['HOME']}/xm/et"

    root_dir = f'{root_dir}/{name_project}'

    # ----------- get sub directory -----------
    keys_exclude.extend(['force_write'])
    split = ','
    # --- add first key
    if key_first is None:
        key_first = list(args_dict.keys())[0]
    keys_exclude.append(key_first)
    sub_dir = args_dict[key_first]
    # --- add keys common
    for key in args_dict.keys():
        if key not in keys_exclude:
            sub_dir += f'{split}{key}={args_dict[key]}'
    sub_dir += ('' if name_suffix == '' else f'{split}{name_suffix}')


    # ----------------- prepare directory ----------
    dirs_full = dict()
    for d_mid in dirs_mid:
        assert d_mid
        tools.makedirs( f'{root_dir}/{d_mid}' )
        dirs_full[d_mid] = f'{root_dir}/{d_mid}/{sub_dir}'
        setattr( args, f'{d_mid}_dir', dirs_full[d_mid] )

    # ----- Move Dirs
    if np.any( [osp.exists( d ) for d in dirs_full.values() ]):  # 如果文件夹存在，则删除
        print(
            f"Exsits sub directory: {sub_dir} in {root_dir} \nMove to discard(y or n)?",
            end='')
        if force_write > 0:
            cmd = 'y'
            print()
        elif force_write < 0:
            exit()
        else:
            cmd = input()
        if cmd == 'y':
            for i in itertools.count():
                if i == 0:
                    suffix = ''
                else:
                    suffix = f'{split}{i}'
                dirs_full_discard = {}
                for d_mid in dirs_mid:
                    tools.mkdir(f'{root_dir}/{d_mid}_discard')
                    dirs_full_discard[d_mid] = f'{root_dir}/{d_mid}_discard/{sub_dir}{suffix}'

                if not np.any( [osp.exists( d ) for d in dirs_full_discard.values() ]):
                    break
            print(tools.colorize(f"Going to discard \n{sub_dir}\n{sub_dir}{suffix}\n"+f"Confirm move(y or n)?", 'red'), end='')
            if force_write > 0:
                cmd = 'y'
                print()
            elif force_write < 0:
                exit()
            else:
                cmd = input()
            if cmd == 'y' \
                    and tools.check_safe_path(args.log_dir, confirm=False):
                import shutil
                for d_mid in dirs_mid:
                    if osp.exists(dirs_full[d_mid]):
                        # shutil.move(  f'Move:\n {dirs_full[d_mid]} \n {dirs_full_discard[d_mid]}','red')
                        shutil.move(dirs_full[d_mid], dirs_full_discard[d_mid])
            else:
                print("Please Rename 'name_suffix'")
                exit()
        else:
            print("Please Rename 'name_suffix'")
            exit()

    for d_mid in dirs_mid:
        tools.mkdir( dirs_full[d_mid] )

    args_str = vars(args)
    with open(f'{args.log_dir}/args.json', 'w') as f:
        json.dump(args_str, f, indent=4, separators=(',', ':'))



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument('--env', help='environment ID', type=str, default='Reacher-v2')
    parser.add_argument('--env', help='environment ID', type=str, default='Swimmer-v2')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num-timesteps', type=int, default=int(1e6))
    parser.add_argument('--play', default=False, action='store_true')
    parser.add_argument('--clipped-type', default='origin', type=str)

    parser.add_argument('--force-write', default=1, type=int)

    args = parser.parse_args()
    args_dict = vars(args)
    args_dict['num_timesteps'] = f"{args_dict['num_timesteps']:.0e}"
    prepare_log( args, args_dict )
    exit()