
import toolsm.logger as logger
from dotmap import DotMap


def tes_logger():
    # dir = "/tmp/a"
    l = logger.Logger(formats='stdout,csv,tensorflow', path='/tmp/', file_basename='aa')
    # l.width_log = [3,4]
    for i in range(10):
        l.log_and_dump_keyvalues(a=1,b=2,c=3)
    for j in range(10):
        l.log_and_dump_keyvalues(a=1,c=3  )
    # l.dumpkvs(1)

    l.close()
    # exit()
tes_logger()
exit()
main_2_sub_2_key_2_grouped_result = logger.group_result_tensorflow_alg_2_env(
  path='/root/d/e/et/DQN/t/initialQ',
  depth=2,
  setting=DotMap(
    __alg={
      'GMQLearning':DotMap(alg='abc')
    },
    fn_load_result=logger._load_csv,
    fn_load_result_kwargs = dict(
      file='process.csv',
      key_x = 'global_step',
      keys_y=['return_','episode'],
      kwargs_readcsv = dict(sep='\t')
    ),
  ),
)
exit()


main_2_sub_2_key_2_grouped_result = logger.get_result_grouped(
  path='/root/d/e/et/DQN/t/initialQ',
  depth=2,
  setting=DotMap(
    groupname_main_setting=DotMap(
      path_inds=[-2]
    ),
    __groupname_main={
      'GMQLearning':DotMap(groupname_main='abc')
    },
    groupname_sub_setting=DotMap(
      json_file='args.json',
      json_keys='env'
    ),
    fn_load_result=logger._load_csv,
    fn_load_result_kwargs = dict(
      file='process.csv',
      key_x = 'global_step',
      keys_y=['return_','episode'],
      kwargs_readcsv = dict(sep='\t')
    ),
  )
)

exit()






fn_get_fn_loaddata = logger.get_load_csv_fn
path_root = '/media/d/e/et/DQN'
task_all = [
    DotMap(
        dir='QLearning_MountainCar',
        key_y_all='return_',
        ylabel_all='Reward',
        __env = {
            'MountainCar': DotMap(legend=DotMap(loc='upper right', fontsize=10))
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
                    task_type='Generate_Result_For_Plot',
                    fn_get_fn_loadresult=fn_get_fn_loaddata,
                    path_root=path_root,
                    algdir_2_setting=algdir_2_setting
                )

logger.gro

logger.write_result_grouped_plot(
                    task_all=task_all,
                    algdir_2_setting=algdir_2_setting,
                    path_root_data=path_root,
                    path_root_save=None,
                    setting_global=DotMap(
                        xlabel='Timesteps',
                        linewidth=1.5,
                        # xticks = DotMap(),#E.G.,div=1e6, unit=r'$\times 10^6$', n=5, round=1
                        ci=60,
                        smooth_window_length=9,
                        fontsize=10,
                        file_ext='png'
                    ),
                    IS_DEBUG=0)

exit()