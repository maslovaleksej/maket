from django.db import models


class Experiment(models.Model):
    type = models.IntegerField(verbose_name='Классификация (0), детекция (1)')

    shortName = models.CharField(max_length=100, verbose_name='Название', unique=True, null=False, blank=False)
    dir_name = models.CharField(max_length=100, verbose_name='Папка', unique=True, null=False, blank=False)

    comments = models.TextField(verbose_name="Описание")


    def __str__(self): return self.shortName

    class Meta:
        verbose_name_plural = '03 Эксперименты'
        verbose_name = 'Эксперимент'
        ordering = ['-shortName']

