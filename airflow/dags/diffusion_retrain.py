from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta
import os
import sys

# Add project to path
sys.path.insert(0, '/opt/airflow')

default_args = {
    'owner': 'sid',
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'email_on_failure': False,
}

with DAG(
    dag_id='diffusion_lifecycle_retrain',
    default_args=default_args,
    description='Full diffusion model retraining pipeline — train 1000 epochs, DPO, quantize, deploy',
    schedule_interval='0 2 * * 0',  # every Sunday at 2am
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=['diffusion', 'mlops', 'nvidia'],
) as dag:

    # ---- Task 1: Validate GPU ----
    validate_gpu = BashOperator(
        task_id='validate_gpu',
        bash_command='python -c "import torch; assert torch.cuda.is_available(), \'GPU not available\'; print(\'GPU OK:\', torch.cuda.get_device_name(0))"',
    )

    # ---- Task 2: Prepare Data ----
    prepare_data = BashOperator(
        task_id='prepare_data',
        bash_command='''
        python -c "
from torchvision import datasets, transforms
transforms_val = transforms.Compose([transforms.ToTensor()])
dataset = datasets.CIFAR10(root='/opt/airflow/outputs/data', train=True, download=True, transform=transforms_val)
print(f'Dataset ready: {len(dataset)} images')
"
        ''',
    )

    # ---- Task 3: Train 1000 Epochs ----
    train_model = BashOperator(
        task_id='train_1000_epochs',
        bash_command='cd /opt/airflow && python -m src.train',
        env={
            'EPOCHS': '1000',
            'BATCH_SIZE': '64',
            'LR': '0.0002',
        },
        execution_timeout=timedelta(hours=12),  # allow up to 12 hours
    )

    # ---- Task 4: Generate + Evaluate Quality ----
    evaluate_quality = BashOperator(
        task_id='evaluate_quality',
        bash_command='cd /opt/airflow && python -m src.score_samples',
    )

    # ---- Task 5: DPO Alignment ----
    run_dpo = BashOperator(
        task_id='run_dpo_alignment',
        bash_command='cd /opt/airflow && python -m src.dpo',
    )

    # ---- Task 6: Quantize ----
    quantize = BashOperator(
        task_id='quantize_model',
        bash_command='cd /opt/airflow && python -m src.quantize',
    )

    # ---- Task 7: Run Benchmark ----
    benchmark = BashOperator(
        task_id='architecture_benchmark',
        bash_command='cd /opt/airflow && python -m src.benchmark',
        execution_timeout=timedelta(hours=4),
    )

    # ---- Task 8: Health Check ----
    health_check = BashOperator(
        task_id='api_health_check',
        bash_command='curl -f http://diffusion-api:8000/health || echo "API not running — deploy manually"',
    )

    # ---- Define order ----
    validate_gpu >> prepare_data >> train_model >> evaluate_quality >> run_dpo >> quantize >> benchmark >> health_check