# Generated by Django 4.1 on 2023-06-19 07:07

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('portal', '0015_experimentattack_batch_num_and_more'),
    ]

    operations = [
        migrations.AlterField(
            model_name='experimentdefense',
            name='eps',
            field=models.CharField(max_length=250, verbose_name='Настройки защиты'),
        ),
        migrations.AlterField(
            model_name='experimentdefense',
            name='eps_name',
            field=models.CharField(default='Сила защиты (eps_def)', max_length=250, verbose_name='Единицы измерения силы защиты'),
        ),
        migrations.CreateModel(
            name='ExperimentResult',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('attack', models.CharField(max_length=100, verbose_name='Атака')),
                ('defense', models.CharField(max_length=100, verbose_name='Защита')),
                ('defense_eps', models.FloatField(verbose_name='Сила защиты')),
                ('param', models.JSONField(blank=True, null=True, verbose_name='param')),
                ('result_json', models.JSONField(verbose_name='result')),
                ('dataset', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='portal.dataset')),
                ('experiment', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='portal.experiment')),
                ('model', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='portal.ins')),
            ],
        ),
    ]
