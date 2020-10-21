import kfp
from kfp import dsl

def extract_data_op():
    return dsl.ContainerOp(
        name='Extract Data',
        image='vyomagg/regression_pipeline_extract_data:latest',
        file_outputs={
            'out_train': '/app/train.pkl',
            'out_test': '/app/test.pkl'
        }
    )


def prepare_op(input_train, input_test):
    return dsl.ContainerOp(
        name='Prepare Data',
        image='vyomagg/regression_pipeline_prepare:latest',
        arguments=[
            '--input_train', input_train,
            '--input_test', input_test
        ],
        file_outputs={
            'out_train': '/app/train_prep.pkl',
            'out_test': '/app/test_prep.pkl'
        }
    )

def train_op(input_train):
    return dsl.ContainerOp(
        name='Train Model',
        image='vyomagg/regression_pipeline_train:latest',
        arguments=[
            '--input_train', input_train
        ],
        file_outputs={
            'train_model': '/app/best.pkl'
        }
    )

def evaluate_op(data_path, model_ckpt_dir):
    return dsl.ContainerOp(
        name='Test Model',
        image='vyomagg/regression_pipeline_evaluate:latest',
        arguments=[
            '--data_path', data_path,
            '--model_ckpt_dir', model_ckpt_dir
        ],
        file_outputs={
            'mean_squared_error':'/app/train_stats_mse.json',
            'root_mean_squared_error': '/app/train_stats_rmse.json',
            'mean_absolute_error':'/app/train_stats_mae.json',
            'r_square_error':'app/train_stats_rsquare.json'
        }
    )


def deploy_model_op(model):
    return dsl.ContainerOp(
        name='Deploy Model',
        image='vyomagg/boston_pipeline_deploy_model:latest',
        arguments=[
            '--model', model
        ]
    )


@dsl.pipeline(
    name='Regression Kubeflow Pipeline',
    description='An example pipeline that trains and logs a regression model.'
)
def regression_pipeline():

    _extract_op = extract_data_op()

    _prepare_op = prepare_op(
        dsl.InputArgumentPath(_extract_op.outputs['out_train']),
        dsl.InputArgumentPath(_extract_op.outputs['out_test'])
    ).after(_extract_op)

    _train_op = train_op(
        dsl.InputArgumentPath(_prepare_op.outputs['out_train'])
    ).after(_prepare_op)

    _evaluate_op = evaluate_op(
        dsl.InputArgumentPath(_prepare_op.outputs['out_test']),
        dsl.InputArgumentPath(_train_op.outputs['train_model'])
    ).after(_train_op)

    deploy_model_op(
        dsl.InputArgumentPath(_train_op.outputs['train_model'])
    ).after(_evaluate_op)


#kfp.compiler.Compiler().compile(regression_pipeline, 'regression_pipeline.zip')

client = kfp.Client()
client.create_run_from_pipeline_func(regression_pipeline, arguments={}, namespace='kubeflow')