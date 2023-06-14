# Generated by Django 4.1 on 2023-06-14 14:50

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('portal', '0006_alter_ins_max_alter_ins_min'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='ins',
            name='arguments',
        ),
        migrations.AddField(
            model_name='ins',
            name='batch_norm_momentum',
            field=models.FloatField(default=1),
            preserve_default=False,
        ),
    ]