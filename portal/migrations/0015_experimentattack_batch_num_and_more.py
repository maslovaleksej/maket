# Generated by Django 4.1 on 2023-06-19 06:42

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('portal', '0014_alter_experimentattack_options_and_more'),
    ]

    operations = [
        migrations.AddField(
            model_name='experimentattack',
            name='batch_num',
            field=models.IntegerField(default=1, verbose_name='Количество пакетов'),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='experimentattack',
            name='batch_size',
            field=models.IntegerField(default=1, verbose_name='Размер пакета'),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='experimentattack',
            name='eps_name',
            field=models.CharField(default='Сила атаки (eps)', max_length=250, verbose_name='Единицы измерения силы атаки'),
        ),
        migrations.AddField(
            model_name='experimentattack',
            name='gpu',
            field=models.BooleanField(default=1, verbose_name='GPU'),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='experimentattack',
            name='save_img',
            field=models.BooleanField(default=1, verbose_name='Save'),
            preserve_default=False,
        ),
        migrations.AlterField(
            model_name='experimentattack',
            name='eps',
            field=models.CharField(max_length=250, verbose_name='Сила атаки'),
        ),
        migrations.CreateModel(
            name='ExperimentDefense',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('shortName', models.CharField(max_length=100, verbose_name='Название')),
                ('eps', models.CharField(max_length=250, verbose_name='Сила атаки')),
                ('eps_name', models.CharField(default='Сила атаки (eps)', max_length=250, verbose_name='Единицы измерения силы атаки')),
                ('save_img', models.BooleanField(verbose_name='Save')),
                ('experiment', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='portal.experiment')),
            ],
            options={
                'verbose_name': 'Эксперимент_защита',
                'verbose_name_plural': '05 Эксперимент_защиты',
                'ordering': ['-shortName'],
            },
        ),
    ]