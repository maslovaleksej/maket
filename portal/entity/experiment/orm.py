from django.db import models

from portal.entity.dataset.orm import Dataset
from portal.entity.model.orm import INS


class Experiment(models.Model):
    shortName = models.CharField(max_length=100, verbose_name='Название', unique=True, null=False, blank=False)
    dir_name = models.CharField(max_length=100, verbose_name='Папка', unique=True, null=False, blank=False)

    comments = models.TextField(verbose_name="Описание")

    def __str__(self): return self.shortName

    class Meta:
        verbose_name_plural = '03 Эксперименты'
        verbose_name = 'Эксперимент'
        ordering = ['-shortName']


class ExperimentAttack(models.Model):
    experiment = models.ForeignKey(Experiment, on_delete=models.CASCADE)
    shortName = models.CharField(max_length=100, verbose_name='Название', unique=False, null=False, blank=False)
    eps = models.CharField(max_length=250, verbose_name='Сила атаки', unique=False, null=False, blank=False)
    eps_name = models.CharField(max_length=250, verbose_name='Единицы измерения силы атаки', unique=False, null=False,
                                blank=False, default='Сила атаки (eps)')
    batch_num = models.IntegerField(verbose_name='Количество пакетов')
    batch_size = models.IntegerField(verbose_name='Размер пакета')
    gpu = models.BooleanField(verbose_name='GPU')
    save_img = models.BooleanField(verbose_name='Save')

    def __str__(self): return self.shortName

    class Meta:
        verbose_name_plural = '04 Эксперимент_атаки'
        verbose_name = 'Эксперимент_атака'
        ordering = ['-shortName']


class ExperimentDefense(models.Model):
    experiment = models.ForeignKey(Experiment, on_delete=models.CASCADE)
    shortName = models.CharField(max_length=100, verbose_name='Название', unique=False, null=False, blank=False)
    eps = models.CharField(max_length=250, verbose_name='Настройки защиты', unique=False, null=False, blank=False)
    eps_name = models.CharField(max_length=250, verbose_name='Единицы измерения силы защиты', unique=False, null=False,
                                blank=False, default='Сила защиты (eps_def)')
    save_img = models.BooleanField(verbose_name='Save')

    def __str__(self): return self.shortName

    class Meta:
        verbose_name_plural = '05 Эксперимент_защиты'
        verbose_name = 'Эксперимент_защита'
        ordering = ['-shortName']


class TrainModelResult(models.Model):
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE)
    model = models.ForeignKey(INS, on_delete=models.CASCADE)
    param = models.JSONField(verbose_name='param', unique=False, null=True, blank=True)
    result_json = models.JSONField(verbose_name='result', unique=False, null=False, blank=False)

    def __str__(self): return self.dataset.shortName

    class Meta:
        verbose_name_plural = '06 Тренированные модели'
        verbose_name = 'Итог тренировки модели'


class ExperimentResult(models.Model):
    experiment = models.ForeignKey(Experiment, on_delete=models.CASCADE)
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE)
    model = models.ForeignKey(INS, on_delete=models.CASCADE)
    attack = models.CharField(max_length=100, verbose_name='Атака', unique=False, null=False, blank=False)
    defense = models.CharField(max_length=100, verbose_name='Защита', unique=False, null=False, blank=False)
    defense_eps = models.FloatField(verbose_name='Сила защиты', unique=False, null=False, blank=False)
    param = models.JSONField(verbose_name='param', unique=False, null=True, blank=True)
    result_json = models.JSONField(verbose_name='result', unique=False, null=False, blank=False)

    def __str__(self): return self.experiment.shortName

    class Meta:
        verbose_name_plural = '07 Результаты экспериментов'
        verbose_name = 'Результат эксперимента'
