from django.contrib import admin

from portal.entity.dataset.orm import Dataset
from portal.entity.experiment.orm import Experiment, ExperimentAttack, ExperimentDefense, TrainModelResult, \
    ExperimentResult
from portal.entity.model.orm import INS


class DatasetAdmin(admin.ModelAdmin):
    list_display = ('shortName',)
    list_display_links = ('shortName',)
    search_fields = ('shortName',)


admin.site.register(Dataset, DatasetAdmin)


class INSAdmin(admin.ModelAdmin):
    list_display = ('shortName', 'type')
    list_display_links = ('shortName',)
    search_fields = ('shortName',)


admin.site.register(INS, INSAdmin)


class ExperimentAdmin(admin.ModelAdmin):
    list_display = ('shortName',)
    list_display_links = ('shortName',)
    search_fields = ('shortName',)


admin.site.register(Experiment, ExperimentAdmin)


class ExperimentAttackAdmin(admin.ModelAdmin):
    list_display = ('shortName', 'eps',)
    list_display_links = ('shortName',)
    search_fields = ('shortName',)


admin.site.register(ExperimentAttack, ExperimentAttackAdmin)


class ExperimentDefenseAdmin(admin.ModelAdmin):
    list_display = ('shortName', 'eps',)
    list_display_links = ('shortName',)
    search_fields = ('shortName',)


admin.site.register(ExperimentDefense, ExperimentDefenseAdmin)


class TrainModelResultAdmin(admin.ModelAdmin):
    list_display = ('dataset', 'model', 'param')
    list_display_links = ('dataset',)
    search_fields = ('dataset',)


admin.site.register(TrainModelResult, TrainModelResultAdmin)


class ExperimentResultAdmin(admin.ModelAdmin):
    list_display = ('experiment', 'dataset', 'model', 'attack', 'defense', 'defense_eps', 'param')
    list_display_links = ('experiment',)
    search_fields = ('experiment',)


admin.site.register(ExperimentResult, ExperimentResultAdmin)