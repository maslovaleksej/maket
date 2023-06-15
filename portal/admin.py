from django.contrib import admin

from portal.entity.dataset.orm import Dataset
from portal.entity.model.orm import INS


class DatasetAdmin(admin.ModelAdmin):
    list_display = ('shortName', )
    list_display_links = ('shortName', )
    search_fields = ('shortName', )

admin.site.register(Dataset, DatasetAdmin)


class INSAdmin(admin.ModelAdmin):
    list_display = ('shortName', 'type')
    list_display_links = ('shortName', )
    search_fields = ('shortName', )

admin.site.register(INS, INSAdmin)


# class ExperimentAdmin(admin.ModelAdmin):
#     list_display = ('shortName', )
#     list_display_links = ('shortName', )
#     search_fields = ('shortName', )
#
# admin.site.register(Experiment, ExperimentAdmin)