from django.db import models


class INS(models.Model):
    type = models.IntegerField(verbose_name='Вектор признаков (0), классификатор(1)')
    dir_name = models.CharField(max_length=100, verbose_name='Имя файла', unique=True, null=False, blank=False)
    shortName = models.CharField(max_length=100, verbose_name='Имя модели', unique=True, null=False, blank=False)
    comments = models.TextField(verbose_name="Описание")
    input_size_x = models.IntegerField(verbose_name='Size X')
    input_size_y = models.IntegerField(verbose_name='Size Y')
    input_size_ch = models.IntegerField(verbose_name='Channels num')
    min = models.IntegerField(verbose_name='Min values input')
    max = models.IntegerField(verbose_name='Max values input')
    batch_norm_momentum = models.FloatField()
    dir_size = models.IntegerField()

    def __str__(self): return self.shortName

    class Meta:
        verbose_name_plural = '02 Модели'
        verbose_name = 'Модель'
        ordering = ['-shortName']

