
import toolsm.logger as logger
from dotmap import DotMap
fn_get_fn_loaddata = logger.get_load_csv_fn
path_root = '/media/d/e/et/DQN'
task_all = [
    DotMap(
        dir='QLearning_MountainCar',
        key_y_all='return_',
        name='return_',
        ylabel='Reward',
        __env = {
            'MountainCar': DotMap(legend=dict(loc='upper right', fontsize=10))
        }
    )
]

algdir_2_setting =  {
	'alg=QLearning,alg_args={n_buffer=4000,n_batch=64,gamma=0.95,keep_Q_other_a=false},policy=Q_Policy_NN,policy_args={epsilon_start=0.1,epsilon_end=0.05,epsilon_decay=2000,Q_initial=null,n_update=2,n_update_target=200,loss=MSELoss}':DotMap(dict(name='Target',pltargs=dict())),
	'alg=QLearning,alg_args={n_buffer=4000,n_batch=64,gamma=0.95,keep_Q_other_a=false},policy=Q_Policy_NN,policy_args={epsilon_start=0.1,epsilon_end=0.05,epsilon_decay=2000,Q_initial=null,n_update=2,n_update_target=null,loss=MSELoss}':DotMap(dict(name='Nothing',pltargs=dict())),
	'alg=QLearning,alg_args={n_buffer=4000,n_batch=64,gamma=0.95,keep_Q_other_a=true},policy=Q_Policy_NN,policy_args={epsilon_start=0.1,epsilon_end=0.05,epsilon_decay=2000,Q_initial=null,n_update=2,n_update_target=null,loss=MSELoss}':DotMap(dict(name='KeepQOfOtherAction',pltargs=dict())),
	'alg=QLearning,alg_args={n_buffer=4000,n_batch=64,gamma=0.95,keep_Q_other_a=true},policy=Q_Policy_NN,policy_args={epsilon_start=0.1,epsilon_end=0.05,epsilon_decay=2000,Q_initial=null,n_update=2,n_update_target=200,loss=MSELoss}':DotMap(dict(name='KeepQOfOtherAction+Target',pltargs=dict()))
}

logger.group_result( task_all,
                    task_type='Generate_Tensorflow',
                    fn_get_fn_loadresult=fn_get_fn_loaddata,
                    path_root=path_root,
                    algdir_2_setting=algdir_2_setting
                    )
exit()

# TODO: document; auto color


logger.plot_result_grouped(task_all=task_all, algdir_2_algsetting=algdir_2_setting, path_root_data=path_root, path_root_save='/root/a', IS_DEBUG=True)

exit()