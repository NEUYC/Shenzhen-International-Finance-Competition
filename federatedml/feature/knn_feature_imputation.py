import numpy as np

from federatedml.model_base import ModelBase
from federatedml.feature.imputer import Imputer
from federatedml.protobuf.generated.knn_feature_imputation_meta2_pb2 import KNNFeatureImputationMeta, KNNFeatureImputerMeta
from federatedml.protobuf.generated.feature_imputation_param_pb2 import FeatureImputationParam
from federatedml.statistic.data_overview import get_header
from federatedml.util import LOGGER
from federatedml.util.io_check import assert_io_num_rows_equal

class FeatureImputation(ModelBase):
    def __init__(self):
        super(FeatureImputation, self).__init__()
        self.summary_obj = None
        self.missing_impute_rate = None
        self.skip_cols = []
        self.cols_replace_method = None
        self.header = None
        from federatedml.param.knn_feature_imputation_param import FeatureImputationParam  #***自修改***
        self.model_param = FeatureImputationParam()

        self.model_param_name = 'FeatureImputationParam'
        self.model_meta_name = 'KNNFeatureImputationMeta'
        self.role = self.role

    def _init_model(self, model_param):
        self.missing_fill_method = model_param.missing_fill_method
        self.col_missing_fill_method = model_param.col_missing_fill_method
        self.default_value = model_param.default_value
        self.n_neighbors = model_param.n_neighbors
        self.missing_impute = model_param.missing_impute
        self.distance = model_param.distance#***自添加***
#zzzzzzzzzzzzzzzzzz
        self.local_only = model_param.local_only #param还未添加
        self.encrypt_param = model_param.encrypt_param
        self.key_length = model_param.key_length
#zzzzzzzzzzzzzzzzzzzzz
    def get_summary(self):
        missing_summary = dict()
        missing_summary["missing_value"] = list(self.missing_impute)#哪些值被认为是缺失
        # missing_summary["missing_impute_value"] = dict(zip(self.header, self.default_value))
        # missing_summary["missing_impute_rate"] = dict(zip(self.header, self.missing_impute_rate))
        # missing_summary["skip_cols"] = self.skip_cols

        return missing_summary

    def load_model(self, model_dict):#无需添加neighbor和distance，model_meta = KNNFeatureImputerMeta()中没有提及任何参数
        param_obj = list(model_dict.get('model').values())[0].get(self.model_param_name)
        meta_obj = list(model_dict.get('model').values())[0].get(self.model_meta_name)
        self.header = param_obj.header
        self.missing_fill, self.missing_fill_method, \
            self.missing_impute, self.n_neighbors, self.distance, self.local_only, self.encrypt_param = load_feature_imputer_model(meta_obj.imputer_meta)

    def save_model(self):#10.14
        meta_obj = save_feature_imputer_model(missing_fill=True,
                                              missing_replace_method=self.missing_fill_method,
                                              missing_impute=self.missing_impute,
                                              n_neighbors=self.n_neighbors,
                                              distance=self.distance,
                                              local_only=self.local_only,
                                              encrypt_param=self.encrypt_param,
                                              key_length=self.key_length)

        return meta_obj

    def export_model(self):
        missing_imputer_meta = self.save_model()
        meta_obj = KNNFeatureImputationMeta(need_run=self.need_run,
                                         imputer_meta=missing_imputer_meta)
        param_obj = FeatureImputationParam(header=self.header)
        model_dict = {
            self.model_meta_name: meta_obj,
            self.model_param_name: param_obj
        }

        return model_dict

    @assert_io_num_rows_equal
    def fit(self, data):
        LOGGER.info(f"Enter Feature Imputation fit")
        imputer_processor = Imputer(self.missing_impute)
        self.header = get_header(data)
        if self.col_missing_fill_method:
            for k in self.col_missing_fill_method.keys():
                if k not in self.header:
                    raise ValueError(f"{k} not found in data header. Please check col_missing_fill_method keys.")
        imputed_data, self.default_value = imputer_processor.fit(data,
                                                                 replace_method=self.missing_fill_method,
                                                                 replace_value=self.default_value,
                                                                 n_neighbor=self.n_neighbors,
                                                                 col_replace_method=self.col_missing_fill_method,
                                                                 distance=self.distance,
                                                                 local_only=self.local_only,
                                                                 encrypt_param=self.encrypt_param,
                                                                 key_length=self.key_length,
                                                                 role=self.role)
        if self.missing_impute is None:
            self.missing_impute = imputer_processor.get_missing_value_list()
        self.missing_impute_rate = imputer_processor.get_impute_rate("fit")
        self.cols_replace_method = imputer_processor.cols_replace_method
        self.skip_cols = imputer_processor.get_skip_cols()
        self.set_summary(self.get_summary())

        return imputed_data#直接换成data

    @assert_io_num_rows_equal
    def transform(self, data):
        LOGGER.info(f"Enter Feature Imputation transform")
        imputer_processor = Imputer(self.missing_impute)
        imputed_data = imputer_processor.transform(data,
                                                   replace_method=self.missing_fill_method,
                                                   replace_value=self.default_value,
                                                   n_neighbor=self.n_neighbors,
                                                   col_replace_method=self.col_missing_fill_method,
                                                   distance=self.distance,
                                                   local_only=self.local_only,
                                                   encrypt_param=self.encrypt_param,
                                                   key_length=self.key_length,
                                                   role=self.role,
                                                   skip_cols=self.skip_cols)
        if self.missing_impute is None:
            self.missing_impute = imputer_processor.get_missing_value_list()

        return imputed_data


def save_feature_imputer_model(missing_fill=False,#不用加self.neighbors和self.distance
                               missing_replace_method=None,
                               missing_impute=None,
                               n_neighbors=None,
                               distance=None,
                               local_only=None,
                               encrypt_param=None,
                               key_length=None):
    model_meta = KNNFeatureImputerMeta()

    model_meta.is_imputer = missing_fill
    if missing_fill:
        if missing_replace_method :
            model_meta.strategy = missing_replace_method

        if missing_impute is not None:
            model_meta.missing_value.extend(map(str, missing_impute))
            model_meta.missing_value_type.extend([type(v).__name__ for v in missing_impute])

        if n_neighbors is not None:
            model_meta.n_neighbors = n_neighbors

        if distance is not None:
            model_meta.distance = distance

        if local_only is not None:
            model_meta.local_only = local_only

        if encrypt_param is not None:
            model_meta.encrypt_param = encrypt_param

        if key_length is not None:
            model_meta.key_length = key_length

    return model_meta


def load_value_to_type(value, value_type):
    if value is None:
        loaded_value = None
    elif value_type in ["int", "int64", "long", "float", "float64", "double"]:
        loaded_value = getattr(np, value_type)(value)
    elif value_type in ["str", "_str"]:
        loaded_value = str(value)
    elif value_type.lower() in ["none", "nonetype"]:
        loaded_value = None
    else:
        raise ValueError(f"unknown value type: {value_type}")
    return loaded_value

def load_feature_imputer_model(model_meta=None):
    missing_fill = model_meta.is_imputer
    missing_replace_method = model_meta.strategy
    n_neighbors = model_meta.n_neighbors
    distance_strategy = model_meta.distance
    local_only = model_meta.local_only
    encrypt_param = model_meta.encrypt_param
    missing_value = list(model_meta.missing_value)
    missing_value_type = list(model_meta.missing_value_type)

    if missing_fill:
        if not n_neighbors :
            n_neighbors = None

        if not distance_strategy :
            distance_strategy = None

        if not encrypt_param :
            encrypt_param = None

        if not local_only :
            local_only = None
        if not missing_replace_method:
            missing_replace_method = None
        #
        if not missing_value:
            missing_value = None
        else:
            missing_value = [load_value_to_type(missing_value[i],
                                                missing_value_type[i]) for i in range(len(missing_value))]

    return missing_fill, missing_replace_method, missing_value, n_neighbors, distance_strategy, local_only, encrypt_param
