from django.db import models


class Dataset(models.Model):
    shortName = models.CharField(max_length=50, verbose_name='Короткое название', unique=True, null=False, blank=False)
    dir_name = models.CharField(max_length=250, verbose_name='Относительный путь', unique=True, null=False, blank=False)
    class_nums = models.IntegerField(verbose_name="Количество классов", unique=False, null=False, blank=False)
    size = models.IntegerField(verbose_name="Количество образцов", unique=False, null=False, blank=False)
    comments = models.TextField(verbose_name="Описание")

    def __str__(self): return self.shortName

    class Meta:
        verbose_name_plural = '01 Датасеты'
        verbose_name = 'Датасет'
        ordering = ['-shortName']

