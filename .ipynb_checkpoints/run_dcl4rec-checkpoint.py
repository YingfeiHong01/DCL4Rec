from recbole.quick_start import run_recbole



parameter_dict = {
   'neg_sampling': None,
}
config_file_list = ['param_ml1m.yaml']
run_recbole(model='DCL4Rec', dataset='ml-1m', config_file_list=config_file_list, config_dict=parameter_dict)
