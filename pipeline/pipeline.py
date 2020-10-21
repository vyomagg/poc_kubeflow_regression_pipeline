import kfp
from kfp import dsl
import yaml


def extract_data_op(train_samples, test_samples):
    return dsl.ContainerOp(
        name='Extract Data',
        image='vyomagg/regression_pipeline_extract_data:latest',
        arguments=[
            '--train_samples', train_samples,
            '--test_samples', test_samples
        ],
        file_outputs={
            'out_train': '/app/data/raw/train.pkl',
            'out_test': '/app/data/raw/test.pkl'
        }
    )


def prepare_op(input_train, input_test, co_relation_threshold):
    return dsl.ContainerOp(
        name='Prepare Data',
        image='vyomagg/regression_pipeline_prepare:latest',
        arguments=[
            '--input_train', input_train,
            '--input_test', input_test,
            '--co_relation_threshold', co_relation_threshold
        ],
        file_outputs={
            'out_train': '/app/data/prepared/train.pkl',
            'out_test': '/app/data/prepared/test.pkl'
        }
    )


def train_op(input_train, fit_intercept , normalize, n_jobs, copy_X):
    return dsl.ContainerOp(
        name='Train Model',
        image='vyomagg/regression_pipeline_train:latest',
        arguments=[
            '--input_train', input_train,
            '--fit_intercept', fit_intercept,
            '--normalize',normalize,
            '--n_jobs',n_jobs,
            '--copy_X', copy_X
        ],
        file_outputs={
            'train_model': '/app/model/Regression_checkpoints/best.pkl'
        }
    )


def evaluate_op(data_path, model_ckpt_dir, metrics):
    return dsl.ContainerOp(
        name='Test Model',
        image='vyomagg/regression_pipeline_evaluate:latest',
        arguments=[
            '--data_path', data_path,
            '--model_ckpt_dir', model_ckpt_dir,
            '--metrics', metrics
        ],
        file_outputs={
            'mean_squared_error': '/app/results/train_stats_mse.json',
            'root_mean_squared_error': '/app/results/train_stats_rmse.json',
            'mean_absolute_error': '/app/results/train_stats_mae.json',
            'r_square_error': 'app/results/train_stats_rsquare.json'
        }
    )


def deploy_model_op(model):
    return dsl.ContainerOp(
        name='Deploy Model',
        image='vyomagg/regression_pipeline_deploy_model:latest',
        arguments=[
            '--model', model
        ]
    )


@dsl.pipeline(
    name='Regression Kubeflow Pipeline',
    description='An example pipeline that trains and logs a regression model.'
)
def regression_pipeline(train_samples: int=466, test_samples: int=50 , co_relation_threshold: float=.5 , fit_intercept: bool=True ,
                        normalize: bool=False, n_jobs: int=2 , copy_X: bool=True, metrics: str="rsquare, mse, rmse, mae" ) :

    _extract_op = extract_data_op(train_samples, test_samples)

    _prepare_op = prepare_op(
        dsl.InputArgumentPath(_extract_op.outputs['out_train']),
        dsl.InputArgumentPath(_extract_op.outputs['out_test']),
        co_relation_threshold
    ).after(_extract_op)

    _train_op = train_op(
        dsl.InputArgumentPath(_prepare_op.outputs['out_train']),
        fit_intercept, normalize, n_jobs, copy_X
    ).after(_prepare_op)

    _evaluate_op = evaluate_op(
        dsl.InputArgumentPath(_prepare_op.outputs['out_test']),
        dsl.InputArgumentPath(_train_op.outputs['train_model']),
        metrics
    ).after(_train_op)

    deploy_model_op(
        dsl.InputArgumentPath(_train_op.outputs['train_model'])
    ).after(_evaluate_op)


#kfp.compiler.Compiler().compile(regression_pipeline, 'regression_pipeline.zip')

## Global Parameters
params = yaml.safe_load(open('params.yaml'))
extract_params = params['extract']
prepare_params = params['prepare']
train_params = params['train']
evaluate_params = params['evaluate']


arguments = { 'train_samples' : extract_params['train_samples'], 'test_samples' : extract_params['test_samples'] ,
            'co_relation_threshold' : prepare_params['co_relation_threshold'] , 'fit_intercept' : train_params['fit_intercept'],
            'normalize' : train_params['normalize'], 'n_jobs' : train_params['n_jobs'] , 'copy_X' : train_params['copy_X'],
            'metrics' : evaluate_params['metrics'] }

## Auto Execution of pipeline
client = kfp.Client(host='http://127.0.0.1:8081', namespace='kubeflow')
client.create_run_from_pipeline_func(regression_pipeline, arguments= arguments)
