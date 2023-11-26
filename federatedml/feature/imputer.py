
import copy
import functools
import numpy as np
import math

from federatedml.feature.fate_element_type import NoneType
from federatedml.feature.instance import Instance
from federatedml.statistic import data_overview
from federatedml.statistic.data_overview import get_header
from federatedml.statistic.statics import MultivariateStatisticalSummary
from federatedml.util import LOGGER

from federatedml.transfer_variable.transfer_class.knn_imputation_transfer_variable import KnnImputationTransferVariable
from federatedml.util import consts
from federatedml.secureprotol import PaillierEncrypt,IpclPaillierEncrypt

class Imputer(object):
    """
    This class provides basic strategies for values replacement. It can be used as missing filled or outlier replace.
    You can use the statistics such as mean, median or max of each column to fill the missing value or replace outlier.
    """

    def __init__(self, missing_value_list=None):
        """
        Parameters
        ----------
        missing_value_list: list, the value to be replaced. Default None, if is None, it will be set to list of blank, none, null and na,
                            which regarded as missing filled. If not, it can be outlier replace, and missing_value_list includes the outlier values
        """
        if missing_value_list is None:
            self.missing_value_list = ['', 'none', 'null', 'na', 'None', np.nan]
        else:
            self.missing_value_list = missing_value_list

        self.abnormal_value_list = copy.deepcopy(self.missing_value_list)#包含所有异常值（如np.nan）的集合,copy.deepcopy方法会创建一个新的Python对象，
        #并将原始列表的所有数据从内存复制到新对象中，因此它们彼此独立存在,由于在Python中对于某些特殊值（如NaN、NoneType）无法直接使用等于号来比较是否相等，
        # 因此通过将self.abnormal_value_list初始化为self.missing_value_list的深拷贝，可以确保两个列表中的元素实质上是相等的，从而避免后续使用时出现错误
        for i, v in enumerate(self.missing_value_list):
            if v != v:#在Python中，NaN不能通过任何操作和比较符号来与其他值相等或不相等。具体而言，对于两个NaN之间的比较，比较结果总是返回False，因为NaN无法与其他数进行比较，也无法被视为正数或负数。
                #在本段代码中，由于数据集可能包含NaN值，因此使用if v != v:语句可以检查是否存在NaN。如果v的值为NaN，那么它与自身比较会得到False，因此才需要这样的判断条件。
                # 该语句的执行结果为True时，代表v等于NaN。如果v等于NaN，接下来的代码将把它替换为np.nan。
                self.missing_value_list[i] = np.nan
                self.abnormal_value_list[i] = NoneType()#self.abnormal_value_list是一个记录异常值的列表，它的作用是保存除缺失值（如空字符串、none、null、na、None和np.nan）之外的需要进行替换的异常值，
                #而不是将其替换为np.nan或其他值。None表示该元素需要进行替换，但目标替换值还未确定，因此这些位置需要在后续中根据具体情况进行处理。
        self.abnormal_value_set = set(self.abnormal_value_list)
        self.support_replace_method = ['min', 'max', 'mean', 'median', 'designated','knn']#可用于替换的所有方法，包括最小值、最大值、平均值、中位数和指定值。#***自添加***
        self.support_output_format = {
            'str': str,
            'float': float,
            'int': int,
            'origin': None
        }

        self.support_replace_area = {#方法的作用范围，不知道all行得通吗？！！！
            'min': 'col',
            'max': 'col',
            'mean': 'col',
            'median': 'col',
            'designated': 'col',
            'knn': 'all'#***自添加***
        }

        self.cols_fit_impute_rate = []
        self.cols_transform_impute_rate = []
        self.cols_replace_method = []
        self.skip_cols = []
        self.transfer_variable = KnnImputationTransferVariable()

    def get_missing_value_list(self):#获取缺失值列表self.missing_value_list
        return self.missing_value_list

    def get_cols_replace_method(self):#获取列所选用的替换方法类型列表self.cols_replace_method
        return self.cols_replace_method

    def get_skip_cols(self):#获取用户指定跳过的列名列表self.skip_cols
        return self.skip_cols

    def get_impute_rate(self, mode="fit"):#获取缺失值占比列表，其中mode参数为fit时获取训练集中每列缺失值占比的列表self.cols_fit_impute_rate
        if mode == "fit":#两种模式的区别，fit是填，transform是换
            return list(self.cols_fit_impute_rate)
        elif mode == "transform":
            return list(self.cols_transform_impute_rate)
        else:
            raise ValueError("Unknown mode of {}".format(mode))

    @staticmethod#fit将数据集中的缺失值替换为转换后的值（说明在transform后），并按照指定的输出格式进行格式化处理，并返回替换后的数据集以及替换过的列索引列表。
    def replace_missing_value_with_cols_transform_value_format(data, transform_list, missing_value_list,
                                                               output_format, skip_cols):
        _data = copy.deepcopy(data)
        replace_cols_index_list = []
        if isinstance(_data, Instance):
            for i, v in enumerate(_data.features):
                if v in missing_value_list and i not in skip_cols:
                    _data.features[i] = output_format(transform_list[i])
                    replace_cols_index_list.append(i)
                else:
                    _data[i] = output_format(v)
        else:
            for i, v in enumerate(_data):
                if str(v) in missing_value_list and i not in skip_cols:
                    _data[i] = output_format(transform_list[i])
                    replace_cols_index_list.append(i)
                else:
                    _data[i] = output_format(v)

        return _data, replace_cols_index_list

    @staticmethod#      fit  根据给定的缺失值异常值集合、变换值列表和需要跳过的列，在数据中寻找缺失值，并将其替换为指定的变换值。
    def replace_missing_value_with_cols_transform_value(data, transform_list, missing_value_list, skip_cols):
        _data = copy.deepcopy(data)
        replace_cols_index_list = []
        if isinstance(_data, Instance):
            new_features = []
            for i, v in enumerate(_data.features):
                if v in missing_value_list and i not in skip_cols:
                    # _data.features[i] = transform_list[i]
                    new_features.append(transform_list[i])
                    replace_cols_index_list.append(i)
                else:
                    new_features.append(v)
            if replace_cols_index_list:
                # new features array will have lowest compatible dtype
                _data.features = np.array(new_features)
        else:
            for i, v in enumerate(_data):
                if str(v) in missing_value_list and i not in skip_cols:
                    _data[i] = str(transform_list[i])
                    replace_cols_index_list.append(i)

        return _data, replace_cols_index_list

    @staticmethod#   transform
    def replace_missing_value_with_replace_value_format(data, replace_value, missing_value_list, output_format):#2023.8.12不知道改不改，transform有而fit没有
        _data = copy.deepcopy(data)
        replace_cols_index_list = []
        if isinstance(_data, Instance):
            for i, v in enumerate(_data.features):
                if v in missing_value_list:
                    _data.features[i] = replace_value
                    replace_cols_index_list.append(i)
                else:
                    _data[i] = output_format(_data[i])
        else:
            for i, v in enumerate(_data):
                if str(v) in missing_value_list:
                    _data[i] = output_format(replace_value)
                    replace_cols_index_list.append(i)
                else:
                    _data[i] = output_format(_data[i])

        return _data, replace_cols_index_list

    @staticmethod#  transform  
    def replace_missing_value_with_replace_value(data, replace_value, missing_value_list):#2023.8.12不知道改不改，transform有而fit没有
        _data = copy.deepcopy(data)
        replace_cols_index_list = []
        if isinstance(_data, Instance):
            new_features = []
            for i, v in enumerate(_data.features):
                if v in missing_value_list:
                    # _data.features[i] = replace_value
                    new_features.append(replace_value)
                    replace_cols_index_list.append(i)
                else:
                    new_features.append(v)
            if replace_cols_index_list:
                # make sure new features array has lowest compatible dtype
                _data.features = np.array(new_features)
        else:
            for i, v in enumerate(_data):
                if str(v) in missing_value_list:
                    _data[i] = str(replace_value)
                    replace_cols_index_list.append(i)

        return _data, replace_cols_index_list

    @staticmethod
    def __get_cols_transform_method(data, replace_method, col_replace_method):#获取需要处理及跳过的列!!!!!
        header = get_header(data)#获取数据的头部信息（列名），如何获取样本（行名）？
        if col_replace_method:
            replace_method_per_col = {col_name: col_replace_method.get(col_name, replace_method) for col_name in header}
        else:
            replace_method_per_col = {col_name: replace_method for col_name in header}
        skip_cols = [v for v in header if replace_method_per_col[v] is None]

        return replace_method_per_col, skip_cols#返回一个元组 (replace_method_per_col, skip_cols)，分别表示每个列的填充方法和跳过的列列表

    def __get_cols_transform_value(self, data, replace_method, replace_value=None):#获取给定数据集每个特征所需的填充值
        """#输出是一个由每个特征对应的填充值组成的列表。

        Parameters
        ----------
        data: input data
        replace_method: dictionary of (column name, replace_method_name) pairs

        Returns
        -------
        list of transform value for each column, length equal to feature count of input data

        """
        summary_obj = MultivariateStatisticalSummary(data, -1, abnormal_list=self.abnormal_value_list)
        header = get_header(data)#如何获取行（样本）？？？
        cols_transform_value = {}
        if isinstance(replace_value, list):
            if len(replace_value) != len(header):
                raise ValueError(
                    f"replace value {replace_value} length does not match with header {header}, please check.")
        for i, feature in enumerate(header):
            if replace_method[feature] is None:
                transform_value = 0
            elif replace_method[feature] == consts.MIN:
                transform_value = summary_obj.get_min()[feature]
            elif replace_method[feature] == consts.MAX:
                transform_value = summary_obj.get_max()[feature]
            elif replace_method[feature] == consts.MEAN:
                transform_value = summary_obj.get_mean()[feature]
            elif replace_method[feature] == consts.MEDIAN:
                transform_value = summary_obj.get_median()[feature]
            elif replace_method[feature] == consts.DESIGNATED:
                if isinstance(replace_value, list):
                    transform_value = replace_value[i]
                else:
                    transform_value = replace_value
                LOGGER.debug(f"replace value for feature {feature} is: {transform_value}")
            elif replace_method[feature] == consts.KNN:#***自添加***
                #transform_value = self.__knn_get_cols_transform_value(data)#还未更改!!!!!!
                transform_value = replace_value#!!!
            else:
                raise ValueError("Unknown replace method:{}".format(replace_method))
            cols_transform_value[feature] = transform_value

        LOGGER.debug(f"cols_transform value is: {cols_transform_value}")
        cols_transform_value = [cols_transform_value[key] for key in header]
        # cols_transform_value = {i: round(cols_transform_value[key], 6) for i, key in enumerate(header)}
        LOGGER.debug(f"cols_transform value is: {cols_transform_value}")
        return cols_transform_value

    @staticmethod
    def _transform_nan(instance):#将实例中的 NaN 值转换为 None。
        feature_shape = instance.features.shape[0]
        new_features = []

        for i in range(feature_shape):
            if instance.features[i] != instance.features[i]:
                # new_features.append(NoneType())
                new_features.append(9999)
            else:
                new_features.append(instance.features[i])
        new_instance = copy.deepcopy(instance)
        new_instance.features = np.array(new_features)
        return new_instance

    def __replace_missing_value_with_knn(self, dataknn, local_only, n_neighbors, distances, missing_value_lists,
                                         encrypt_param, key_length,role):
        def computer_distance(instance1, instance2, dis):
            if dis == 'euclideandistance':
                result = euclidean_distance(instance1, instance2)
            elif dis == 'manhattandistance':
                result = manhattan_distance(instance1, instance2)
            else:
                raise ValueError(
                    "impute {} not supported, should be one of distance".format(dis)
                )
            return result

        def euclidean_distance(instance1, instance2):
            # 检查输入数组长度是否相等
            if len(instance1) != len(instance2):
                raise ValueError("输入数组长度不相等")
            # 计算差值平方和
            diff_squared = np.sum(np.power(instance1 - instance2, 2)) / len(instance1)
            # 计算欧式距离
            gap = np.sqrt(diff_squared)
            return gap

        def manhattan_distance(instance1, instance2):
            # 检查输入数组长度是否相等
            if len(instance1) != len(instance2):
                raise ValueError("输入数组长度不相等")
            # 计算差值绝对值和
            diff_abs = np.sum(np.abs(instance1 - instance2)) / len(instance1)
            # 计算曼哈顿距离
            gap = diff_abs
            return gap

        def remove_abnormal_values(arr1, arr2, abnormal_list):
            normal1 = []
            normal2 = []
            for i in range(len(arr1)):
                if arr1[i] in abnormal_list or arr2[i] in abnormal_list:
                    continue
                normal1.append(arr1[i])
                normal2.append(arr2[i])
            return np.array(normal1), np.array(normal2)

        def extract_elements(a, b):
            return [b[i] for i in a]

        def fanghui(np_train_data, id_list, data):  # 这里必须是反这的，因为输入的参数先输入填充好的后才是table
            fanghui1 = []
            for i, v in data:
                v_index = id_list.index(int(i))
                v.features = np_train_data[v_index]  # i是str的数据，需要转为int才能使用
                fanghui1.append((i, v))
            return fanghui1

        schema = dataknn.schema

        # 生成密钥并将公钥发给对方
        if  local_only:
            replace_cols_index_list = []
            arr = []
            id_list = []
            for i, instance_i in enumerate(dataknn.collect()):  # 遍历每一个样本
                id_list.append(int(instance_i[0]))
                for j, feature_value_j in enumerate(instance_i[1].features):  # 遍历当前样本特征值
                    if feature_value_j in missing_value_lists:  # 判断是否有缺失值
                        temp_differences = []  # 临时列表用于存储样本距离
                        temp_save_id = []  # 临时列表用于存储被计算的样本ID
                        for m, instance_m in enumerate(dataknn.collect()):  # 遍历其它无缺失样本
                            if m != i and (not instance_m[1].features[j] in missing_value_lists):  # 遍历其它无缺失样本
                                nor_arry1, nor_arry2 = remove_abnormal_values(instance_i[1].features,  # 去掉异常值
                                                                              instance_m[1].features,
                                                                              missing_value_lists)
                                difference = computer_distance(nor_arry1, nor_arry2, distances)
                                temp_differences.append(difference)  # 存储样本距离
                                temp_save_id.append(m)  # 存储被计算的样本索引

                        # 找到最近的id列表nearest_ids_row
                        sorted_indices = np.argsort(temp_differences)  # 对距离进行排序，并获取索引
                        nearest_indices = sorted_indices[:n_neighbors]  # 取前n_neighbor个最小距离的索引
                        # nearest_ids_row = temp_save_id[nearest_indices].tolist()  # 根据索引获取对应的ID值
                        nearest_ids_row = extract_elements(nearest_indices, temp_save_id)

                        # 根据列表求均值
                        total_difference = 0
                        for n, instance_n in enumerate(dataknn.collect()):  # 遍历每一个样本
                            if n in nearest_ids_row:
                                total_difference += instance_n[1].features[j]

                        impute_value = total_difference / n_neighbors
                        formatted_num = "{:.6f}".format(impute_value)
                        # 插补
                        instance_i[1].features[j] = formatted_num
                        # 记录处理过的列
                        replace_cols_index_list.append(j)
                        replace_cols_index_list = list(set(replace_cols_index_list))

                    if j == len(instance_i[1].features) - 1:
                        arr.append(instance_i[1].features)
            # 放回
            _data = dataknn.copy()
            array = np.array(arr)
            f = functools.partial(fanghui, array, id_list)
            data_table = _data.mapPartitions(f, use_previous_behavior=False)
            data_table.schema = schema

        if not local_only:
            def computer_local_distance(instance1, instance2, dis):
                if dis == 'euclideandistance':
                    result = euclidean_local_distance(instance1, instance2)
                elif dis == 'manhattandistance':
                    result = manhattan_local_distance(instance1, instance2)
                else:
                    raise ValueError(
                        "impute {} not supported, should be one of distance".format(dis)
                    )
                return result

            def euclidean_local_distance(instance1, instance2):
                # 检查输入数组长度是否相等
                if len(instance1) != len(instance2):
                    raise ValueError("输入数组长度不相等")
                # 计算差值平方和
                gap = np.sum(np.power(instance1 - instance2, 2)) / len(instance1)
                # 计算欧式距离
                # gap = np.sqrt(diff_squared)
                return gap

            def manhattan_local_distance(instance1, instance2):
                # 检查输入数组长度是否相等
                if len(instance1) != len(instance2):
                    raise ValueError("输入数组长度不相等")
                # 计算差值绝对值和
                diff_abs = np.sum(np.abs(instance1 - instance2)) / len(instance1)
                # 计算曼哈顿距离
                gap = diff_abs
                return gap

            def generate_encrypter(key_length, encrypt_param):
                LOGGER.info("generate encrypter")
                if encrypt_param == 'paillier':
                    encrypter = PaillierEncrypt()
                    encrypter.generate_key(n_length=key_length)
                elif encrypt_param == 'ipcl':
                    encrypter = IpclPaillierEncrypt()
                    encrypter.generate_key(n_length=key_length)
                else:
                    raise NotImplementedError("unknown encrypt Ftype {}".format(encrypt_param.lower()))
                return encrypter

            def dis_all(data):#使用对称方式计算全距离矩阵
                # array_zeros = np.zeros((data.shape[0], data.shape[0]))
                array_zeros = [[0] * data.shape[0] for _ in range(data.shape[0])]
                for m in range(data.shape[0] - 1):
                    for n in range(m + 1, data.shape[0]):
                        nor_arry1, nor_arry2 = remove_abnormal_values(data[m],data[n],missing_value_lists)
                        array_zeros[m][n] = computer_local_distance(nor_arry1, nor_arry2, distances)
                        array_zeros[n][m] = array_zeros[m][n]
                return array_zeros

            def custom_sort_key(item):#将距离为0的距离改为无穷大
                value, name = item
                if np.isnan(value):
                    return (float('inf'), name)  # 使用特别大的值代表NaN
                return (value, name)

            def get_features(data):
                features = []
                features_num = data.count()
                # a = data.first()
                # features_or = data.take(features_num)
                features_or = list(data.collect())  # 取出每一行
                for i in range(features_num):
                    features.append(features_or[i][1].features.tolist())
                return np.array(features).reshape(features_num, data.first()[1].features.size)

            if encrypt_param != None:
                encrypter = generate_encrypter(key_length, encrypt_param)  # 生成密钥,rsa_bit=1024
                if encrypt_param != 'rsa':
                    pub_key = encrypter.get_public_key()  # 获取公钥
                    privacy_key = encrypter.get_privacy_key()
                if role == consts.GUEST:  # remot和get不会用
                    self.transfer_variable.pub_key_guest.remote(pub_key, role=consts.HOST, idx=-1)  # 发送本地idx给对方
                    self.pub_key_get = self.transfer_variable.pub_key_host.get(idx=0)  ##获取对方idx
                if role == consts.HOST:
                    self.transfer_variable.pub_key_host.remote(pub_key, role=consts.GUEST, idx=-1)  # 发送本地idx给对方
                    self.pub_key_get = self.transfer_variable.pub_key_guest.get(idx=0)  ##获取对方idx
            else:
                pass

            replace_cols_index_list = []
            id_list = []
            for i, instance_i in enumerate(dataknn.collect()):  # 遍历每一个样本
                id_list.append(int(instance_i[0]))

            #从dataknn中提取特征数据并转化为np格式
            np_train_data = get_features(dataknn)
            #获取缺失索引
            missing_indices = np.argwhere(np_train_data == 9999)
            #计算距离矩阵
            distances_all = dis_all(np_train_data)

            #去除nan值
            for i in range(len(distances_all)):
                for j in range(len(distances_all[i])):
                    if math.isnan(distances_all[i][j]):
                        distances_all[i][j] = 0.0
            # 可选加密
            if encrypt_param != None:
                en_distance_remove = copy.deepcopy(distances_all)
                for i in range(len(distances_all)):
                    for j in range(len(distances_all[i])):
                        en_distance_remove[i][j] = self.pub_key_get.encrypt(distances_all[i][j])
            else:
                en_distance_remove = distances_all

            if role == consts.GUEST:  # remot和get不会用
                self.transfer_variable.distance_remove_guest.remote(en_distance_remove, role=consts.HOST, idx=-1,
                                                                    suffix=())  # 发送本地idx给对方
                remote_local_distance = self.transfer_variable.distance_remove_host.get(idx=0, suffix=())  ##获取对方idx,ost???????改idx_host
            if role == consts.HOST:
                self.transfer_variable.distance_remove_host.remote(en_distance_remove, role=consts.GUEST, idx=-1,
                                                                   suffix=())  # 发送本地idx给对方
                remote_local_distance = self.transfer_variable.distance_remove_guest.get(idx=0, suffix=())  ##获取对方idx

            common_distance = np.array(remote_local_distance) + np.array(distances_all)

            if encrypt_param != None:  # 可选解密
                de_dis_tol = common_distance.copy()
                for i in range(len(common_distance)):
                    for j in range(len(common_distance[i])):
                        de_dis_tol[i][j] = privacy_key.decrypt(common_distance[i][j])
                dis_tol = np.around(np.sqrt(np.array(de_dis_tol,dtype=np.float32)), decimals=4)

            else:
                dis_tol = np.around(np.sqrt(common_distance), decimals=4)

            imputed_data = copy.deepcopy(np_train_data)
            for i, j in missing_indices:
                distances = []
                for m in range(np_train_data.shape[0]):
                    if m != i and (not np_train_data[m, j] in missing_value_lists):
                        distances.append((dis_tol[i][m], np_train_data[m, j]))
                distances.sort(key=custom_sort_key)
                imputed_data[i, j] = np.mean([x[1] for x in distances[:n_neighbors]])
                replace_cols_index_list.append(j)
                replace_cols_index_list = list(set(replace_cols_index_list))

            _data = dataknn.copy()
            array = np.array(imputed_data)
            f = functools.partial(fanghui, array, id_list)
            data_table = _data.mapPartitions(f, use_previous_behavior=False)
            data_table.schema = schema

        return data_table, replace_cols_index_list

    def __fit_replace(self, data, replace_method, replace_value=None, output_format=None, col_replace_method=None,
                      n_neighbors=None, distance=None, missing_value_list=None,local_only=None,encrypt_param=None,key_length=None,role=None):  # ***自添加***
        replace_method_per_col, skip_cols = self.__get_cols_transform_method(data, replace_method, col_replace_method)

        schema = data.schema#填补后给

        if isinstance(data.first()[1], Instance):
            data = data.mapValues(lambda v: Imputer._transform_nan(v))#是不是map函数后面都要接data.schema = schema？
            data.schema = schema

        cols_transform_value = self.__get_cols_transform_value(data, replace_method_per_col,
                                                               replace_value=replace_value)
        self.skip_cols = skip_cols
        skip_cols = [get_header(data).index(v) for v in skip_cols]
        if output_format is not None:
            f = functools.partial(Imputer.replace_missing_value_with_cols_transform_value_format,
                                  # 其作用是对一个函数偏函数化（部分应用），即固定函数中的某些参数，并返回一个新函数。
                                  transform_list=cols_transform_value, missing_value_list=self.abnormal_value_set,
                                  # missing_value_list用于标识缺失值的异常值集合。
                                  output_format=output_format, skip_cols=set(skip_cols))  # ***自修改***
        else:
            f = functools.partial(Imputer.replace_missing_value_with_cols_transform_value,
                                  transform_list=cols_transform_value, missing_value_list=self.abnormal_value_set,
                                  skip_cols=set(skip_cols))  # ***自修改***

        if replace_method == 'knn':
            transform_data, cols_transform_value = self.__replace_missing_value_with_knn(data, local_only, n_neighbors,
                                                                                  distance, missing_value_list, encrypt_param, key_length,role)
            self.cols_replace_method = replace_method_per_col
        else:
            process_data = data.mapValues(f)  # 将RDD的每一个value都应用函数f进行处理，KEY不变，返回一个新的RDD。#这里的value就是instance
            transform_data = process_data.mapValues(lambda v: v[0])
            self.cols_replace_method = replace_method_per_col
            LOGGER.info(
                "finish replace missing value with cols transform value, replace method is {}".format(replace_method))
        return transform_data, cols_transform_value

    def __transform_replace(self, data, replace_method, replace_value=None, replace_area='col',output_format=None, col_replace_method=None,skip_cols=None,
                      n_neighbors=None, distance=None, missing_value_list=None,local_only=None,encrypt_param=None,key_length=None,role=None):#基于指定的缺失值替换信息，使用mapValues()函数将数据集中的缺失值替换为指定值或按列替换。
        replace_method_per_col, skip_cols = self.__get_cols_transform_method(data, replace_method, col_replace_method)#初版knn跑min出错，'int' object is not subscriptable修改前
        skip_cols = [get_header(data).index(v) for v in skip_cols]

        schema = data.schema
        if isinstance(data.first()[1], Instance):
            data = data.mapValues(lambda v: Imputer._transform_nan(v))
            data.schema = schema

        # self.skip_cols = skip_cols
        # skip_cols = [get_header(data).index(v) for v in skip_cols]#'int' object is not subscriptable修改前
        cols_transform_value = self.__get_cols_transform_value(data, replace_method_per_col,
                                                               replace_value=replace_value)

        if replace_area == 'all':
            if output_format is not None:
                f = functools.partial(Imputer.replace_missing_value_with_replace_value_format,
                                      replace_value=replace_value, missing_value_list=self.abnormal_value_set,
                                      output_format=output_format)
            else:
                f = functools.partial(Imputer.replace_missing_value_with_replace_value,
                                      replace_value=replace_value, missing_value_list=self.abnormal_value_set)
        elif replace_area == 'col':
            if output_format is not None:
                f = functools.partial(Imputer.replace_missing_value_with_cols_transform_value_format,
                                      transform_list=cols_transform_value, missing_value_list=self.abnormal_value_set,
                                      output_format=output_format,
                                      skip_cols=set(skip_cols))
            else:
                f = functools.partial(Imputer.replace_missing_value_with_cols_transform_value,
                                      transform_list=cols_transform_value, missing_value_list=self.abnormal_value_set,
                                      skip_cols=set(skip_cols))
        else:
            raise ValueError("Unknown replace area {} in Imputer".format(replace_area))

        # if output_format is not None:
        #     f = functools.partial(Imputer.replace_missing_value_with_cols_transform_value_format,
        #                           transform_list=replace_value, missing_value_list=self.abnormal_value_set,
        #                           output_format=output_format,
        #                           skip_cols=set(skip_cols))
        # else:
        #     f = functools.partial(Imputer.replace_missing_value_with_cols_transform_value,
        #                           transform_list=replace_value, missing_value_list=self.abnormal_value_set,
        #                           skip_cols=set(skip_cols))

        if replace_method == 'knn':
            transform_data, cols_transform_value = self.__replace_missing_value_with_knn(data, local_only, n_neighbors,
                                                                                         distance, missing_value_list,
                                                                                         encrypt_param, key_length,
                                                                                         role)
        else:
            # process_data = data.mapValues(f)  # 将RDD的每一个value都应用函数f进行处理，KEY不变，返回一个新的RDD。#这里的value就是instance
            transform_data = data.mapValues(f)
            # transform_data = process_data.mapValues(lambda v: v[0])
            LOGGER.info(
                "finish replace missing value with cols transform value, replace method is {}".format(replace_method))

        return transform_data

    @staticmethod
    def __get_impute_number(some_data):#统计某个数据集中每个特征（或样本）上缺失值的数量以及数据集中的总缺失值数量。
        impute_num_list = None
        data_size = None

        for line in some_data:
            processed_data = line[1][0]
            index_list = line[1][1]
            if not data_size:
                if isinstance(processed_data, Instance):
                    data_size = data_overview.get_instance_shape(processed_data)
                else:
                    data_size = len(processed_data)
                # data_size + 1, the last element of impute_num_list used to count the number of "some_data"
                impute_num_list = [0 for _ in range(data_size + 1)]

            impute_num_list[data_size] += 1
            for index in index_list:
                impute_num_list[index] += 1

        return np.array(impute_num_list)

    def __get_impute_rate_from_replace_data(self, data):
        impute_number_statics = data.applyPartitions(self.__get_impute_number).reduce(lambda x, y: x + y)
        cols_impute_rate = impute_number_statics[:-1] / impute_number_statics[-1]

        return cols_impute_rate

    def fit(self, data, replace_method=None, replace_value=None, n_neighbor=1, col_replace_method=None,
             distance=None, local_only=None, encrypt_param=None, key_length=None, role=None,output_format=consts.ORIGIN):
        """
        Apply imputer for input data
        Parameters
        ----------
        data: Table, each data's value should be list
        replace_method: str, the strategy of imputer, like min, max, mean or designated and so on. Default None
        replace_value: str, if replace_method is designated, you should assign the replace_value which will be used to replace the value in imputer_value_list
        output_format: str, the output data format. The output data can be 'str', 'int', 'float'. Default origin, the original format as input data
        col_replace_method: dict of (col_name, replace_method), any col_name not included will take replace_method
        n_neighbor
        distance

        Returns
        ----------
        fit_data:data_instance, data after imputer
        cols_transform_value: list, the replace value in each column
        """
        if output_format not in self.support_output_format:
            raise ValueError("Unsupport output_format:{}".format(output_format))

        output_format = self.support_output_format[output_format]

        if isinstance(replace_method, str):#如果它是字符串类型，则将其转换为小写，并检查它是否在支持的填充方法中。
            replace_method = replace_method.lower()
            if replace_method not in self.support_replace_method:
                raise ValueError("Unknown replace method:{}".format(replace_method))
        elif replace_method is None and col_replace_method is None:#根据训练数据的第一个实例的数据类型来设置默认替换值；如果第一个实例的数据类型是 Instance，则将值设为 0，否则将值设为字符串 '0'；
            if isinstance(data.first()[1], Instance):
                replace_value = 0
            else:
                replace_value = '0'
        elif replace_method is None and col_replace_method is not None:
            LOGGER.debug(f"perform computation on selected cols only: {col_replace_method}")#只对选择的列进行计算
        else:
            raise ValueError("parameter replace_method should be str or None only")
        if isinstance(col_replace_method, dict):#是否是一个字典，并检查其中每个键值对表示的列填充方法是否在支持的范围内。
            for col_name, method in col_replace_method.items():
                method = method.lower()
                if method not in self.support_replace_method:
                    raise ValueError("Unknown replace method:{}".format(method))
                col_replace_method[col_name] = method

        #dfjhgjhdghdsgklsdjgksdjglkajkdjgkdsjg
        # n_neighbor+=1

        process_data, cols_transform_value = self.__fit_replace(data, replace_method, replace_value, output_format,
                                                                col_replace_method,
                                                                n_neighbors=n_neighbor,
                                                                distance=distance,
                                                                missing_value_list=self.missing_value_list,
                                                                local_only=local_only,
                                                                encrypt_param=encrypt_param,
                                                                key_length=key_length,
                                                                role=role)

        # self.cols_fit_impute_rate = self.__get_impute_rate_from_replace_data(process_data)
        # process_data = process_data.mapValues(lambda v: v[0])
        process_data.schema = data.schema

        return process_data, cols_transform_value

    def transform(self, data, replace_method=None, replace_value=None, n_neighbor=1, col_replace_method=None,
             distance=None, local_only=None, encrypt_param=None, key_length=None, role=None,output_format=consts.ORIGIN,skip_cols=None):#用于对新的数据应用已经计算好的缺失值处理器
        """
        Transform input data using Imputer with fit results#将输入数据集中的缺失值进行填充和替换
        Parameters
        ----------
        data: Table, each data's value should be list
        replace_value:
        output_format: str, the output data format. The output data can be 'str', 'int', 'float'. Default origin, the original format as input data

        Returns
        ----------
        transform_data:data_instance, data after transform
        """
        if output_format not in self.support_output_format:
            raise ValueError("Unsupport output_format:{}".format(output_format))

        output_format = self.support_output_format[output_format]
        skip_cols = [] if skip_cols is None else skip_cols

        # Now all of replace_method is "col", remain replace_area temporarily
        # replace_area = self.support_replace_area[replace_method]
        replace_area = "col"
        process_data = self.__transform_replace(data, replace_method, replace_value, replace_area, output_format,#10.30
                                                                col_replace_method,
                                                                skip_cols=skip_cols,
                                                                n_neighbors=n_neighbor,
                                                                distance=distance,
                                                                missing_value_list=self.missing_value_list,
                                                                local_only=local_only,
                                                                encrypt_param=encrypt_param,
                                                                key_length=key_length,
                                                                role=role)
        # self.cols_transform_impute_rate = self.__get_impute_rate_from_replace_data(process_data)
        if replace_method != 'knn':
            process_data = process_data.mapValues(lambda v: v[0])
        process_data.schema = data.schema

        return process_data

