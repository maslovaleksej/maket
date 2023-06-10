from django.db import models


class INS(models.Model):
    shortName = models.CharField(max_length=50, verbose_name='Короткое название', unique=True, null=False, blank=False)
    dir_name = models.CharField(max_length=250, verbose_name='Имя папки', unique=True, null=False, blank=False)

    def __str__(self): return self.shortName

    class Meta:
        verbose_name_plural = '02 Модели'
        verbose_name = 'Модель'
        ordering = ['-shortName']

#
# class Experiment(models.Model):
#     shortName = models.CharField(max_length=50, verbose_name='Короткое название', unique=True, null=False, blank=False)
#     dir_path = models.CharField(max_length=250, verbose_name='Относительный путь', unique=True, null=False, blank=False)
#     ins = models.ManyToManyField(INS, verbose_name="модели", related_name="модели")
#     dataset = models.ForeignKey(Dataset, on_delete=models.RESTRICT, null=False, blank=False, related_name='Датасет', verbose_name='Датасет')
#
#
#     def __str__(self): return self.shortName
#
#
#     class Meta:
#         verbose_name_plural = '03 Эксперименты'
#         verbose_name = 'Эксперимент'
#         ordering = ['-shortName']
#
