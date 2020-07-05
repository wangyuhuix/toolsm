
import toolsm.logger as logger
from dotmap import DotMap
fn_get_fn_loaddata = logger.get_load_csv_fn
path_root = '/media/d/e/et/DQN'
task_all = [
    DotMap(
        dir='QLearning_MountainCar',
        key_y_all=['return_'],
        name='return_',
        ylabel='Reward',
        env = {
            'MountainCar': DotMap(legend=dict(loc='upper right', fontsize=10))
        }
    )
]


# logger.group_result( task_all,
#                     'Generate_Result_For_Plot',
#                     fn_get_fn_loaddata,
#                     path_root=path_root
#                     )
# exit()

# TODO: document; auto color

algdir_2_algsetting =  {
	'alg=QLearning,alg_args={n_buffer=4000,n_batch=64,gamma=0.95,keep_Q_other_a=false},policy=Q_Policy_NN,policy_args={epsilon_start=0.1,epsilon_end=0.05,epsilon_decay=2000,Q_initial=null,n_update=2,n_update_target=200,loss=MSELoss}':DotMap(dict(name='Target',pltargs=dict())),
	'alg=QLearning,alg_args={n_buffer=4000,n_batch=64,gamma=0.95,keep_Q_other_a=false},policy=Q_Policy_NN,policy_args={epsilon_start=0.1,epsilon_end=0.05,epsilon_decay=2000,Q_initial=null,n_update=2,n_update_target=null,loss=MSELoss}':DotMap(dict(name='Nothing',pltargs=dict())),
	'alg=QLearning,alg_args={n_buffer=4000,n_batch=64,gamma=0.95,keep_Q_other_a=true},policy=Q_Policy_NN,policy_args={epsilon_start=0.1,epsilon_end=0.05,epsilon_decay=2000,Q_initial=null,n_update=2,n_update_target=null,loss=MSELoss}':DotMap(dict(name='KeepQOfOtherAction',pltargs=dict())),
	'alg=QLearning,alg_args={n_buffer=4000,n_batch=64,gamma=0.95,keep_Q_other_a=true},policy=Q_Policy_NN,policy_args={epsilon_start=0.1,epsilon_end=0.05,epsilon_decay=2000,Q_initial=null,n_update=2,n_update_target=200,loss=MSELoss}':DotMap(dict(name='KeepQOfOtherAction+Target',pltargs=dict()))
}

logger.plot_result_grouped(task_all=task_all, algdir_2_setting=algdir_2_algsetting, path_root_data=path_root, path_root_save='/root/a', IS_DEBUG=True)

exit()



alg_setting_all = {

    'PPO-RB': DotMap(dict( pltargs=dict(color='green'))),
    'TR-PPO': DotMap(dict( pltargs=dict(color='darkviolet'))),
    'Truly PPO': DotMap(dict( pltargs=dict(color='blue'))),
    'TRPO': DotMap(dict( pltargs=dict(color='lawngreen'))),
    'PPO-penalty': DotMap(dict( pltargs=dict(color='brown'))),

    'SAC': DotMap(dict(pltargs=dict(color='orange', zorder=-1))),
    'TD3': DotMap(dict(pltargs=dict(color='deepskyblue', zorder=-2))),  # deepskyblue

    'PPO': DotMap(dict( pltargs=dict(color='red'))),


    'PPO-0.6': DotMap(dict( pltargs=dict(color='hotpink', linestyle='--'))),
    'A2C': DotMap(dict( pltargs=dict(color='grey', linestyle='--'))),
    'PPO-direct': DotMap(dict( pltargs=dict(color='mediumvioletred', linestyle=':'))),
    'TR-PPO-direct': DotMap(dict( pltargs=dict(color='mediumpurple', linestyle=':'))),
}
fontsize = 14

ylim_end_approxkl = 0.4
task_setting_all = [
DotMap(
dict(
    name= 'eprewmean',
    dir = 'eprewmean_atari',
    ylabel= 'Reward',
    env = {
        # "PongNoFrameskip":DotMap(),
        # "SeaquestNoFrameskip": DotMap(ylim_start=200, ylim_end=2200),
        # "BreakoutNoFrameskip": DotMap(ylim_start=0, ylim_end=420, legend=dict(loc='upper left', fontsize=16)),
        # "EnduroNoFrameskip": DotMap(ylim_start=0, ylim_end=800),
        # "QbertNoFrameskip": DotMap(),
        # "AtlantisNoFrameskip": DotMap(ylim_start=0, ylim_end=2400000),
        # "SpaceInvadersNoFrameskip":DotMap(ylim_start=100, ylim_end=1100),
        "BeamRiderNoFrameskip": DotMap(ylim_start=250, ylim_end=3000),
        # 'BattleZoneNoFrameskip':DotMap()
    }
)),
# DotMap(
#     dict(
#     name= 'eprewmean_eval',
#     ylabel= 'Reward',
#     env = {
#         "HalfCheetah": DotMap(ylim_start=0, ylim_end=11000,legend=dict(loc='upper right')),
#         "Humanoid": DotMap(
#             xticks_setting = DotMap( round=0),
#             __alg={
#                 'SAC':DotMap( window_length=41 ),
#                 'TD3':DotMap( window_length=21 ),
#             }
#         ),
#         "Reacher": DotMap(ylim_start=-18, ylim_end=-3),
#         "Hopper": DotMap(),
#         "Swimmer": DotMap(ylim_start=0),
#         "Walker2d": DotMap(ci=40),
#     }
# )),
]