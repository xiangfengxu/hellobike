'''
@author: xuxiangfeng 
@date: 2020/2/5
'''

from typing import List
from ortools.constraint_solver import routing_enums_pb2, pywrapcp
from math import sqrt, asin, sin, cos, radians
import numpy as np
from flask import Flask, request, jsonify
import copy
import logging


class MyLogger:

    @staticmethod
    def get_log(log_file_name: str, log_level: str, logger_name: str = None):
        '''
        log_file_name: 日志文件的路径，例: ../code/mylog.txt
        log_level: 日志记录的等级，['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'NOTSET']
        '''
        # 创建一个logger
        logger = logging.getLogger(logger_name)

        # 指定日志的最低输出级别，默认为WARN级别
        logger.setLevel(log_level)

        # 创建一个handler用于写入日志文件
        file_handler = logging.FileHandler(log_file_name)

        # 创建一个handler用于输出控制台
        # console_handler = logging.StreamHandler()

        # 定义handler的输出格式
        formatter = logging.Formatter(
            '[%(asctime)s]-[%(filename)s line: %(lineno)d]-%(levelname)s: %(message)s')
        file_handler.setFormatter(formatter)
        # console_handler.setFormatter(formatter)

        # 给logger添加handler
        logger.addHandler(file_handler)
        # self.__logger.addHandler(console_handler)

        return logger


app = Flask(__name__)
log = MyLogger.get_log(log_file_name='log.txt', log_level="INFO")


class Distance:
    @staticmethod
    def _haversine(startPos, endPos) -> int:
        startLng, startLat = startPos
        endLng, endLat = endPos
        startLng, startLat, endLng, endLat = map(radians, [startLng, startLat, endLng, endLat])
        dlon = endLng - startLng
        dlat = endLat - startLat
        a = sin(dlat / 2) ** 2 + cos(startLat) * cos(endLat) * sin(dlon / 2) ** 2
        c = 2 * asin(sqrt(a))
        r = 6371  # Radius of earth in kilometers. Use 3956 for miles
        return round(c * r * 1000)  # 单位：米

    @staticmethod
    def manhattanDistance(node1, node2) -> int:
        '''
        两点之间的曼哈顿距离
        '''
        middlePos = (node1.position.lng, node2.position.lat)
        startPos = (node1.position.lng, node1.position.lat)
        endPos = (node2.position.lng, node2.position.lat)
        dis = Distance._haversine(startPos, middlePos) + Distance._haversine(endPos, middlePos)
        return dis

    @staticmethod
    def lineDistance(node1, node2) -> int:
        '''
        两点之间的直线距离
        '''
        startPos = (node1.position.lng, node1.position.lat)
        endPos = (node2.position.lng, node2.position.lat)
        dis = Distance._haversine(startPos, endPos)
        return dis


class Position:
    def __init__(self, lng, lat):
        self.lng = lng
        self.lat = lat


class BatterySite:
    def __init__(self, lng, lat):
        self.position = Position(lng, lat)


class ExchangeBatteryUser:
    def __init__(self, batteryNum: int, lng: float, lat: float, vehicleCapacity: int, changeBatterySpeed=30,
                 rideSpeed=15):
        self.batteryNum = batteryNum
        self.position = Position(lng, lat)
        self.vehicleCapacity = vehicleCapacity
        self.changeBatterySpeed = changeBatterySpeed  # 换电池的速度， 秒/块
        self.rideSpeed = rideSpeed  # km/h


class PowerExchangeTask:
    def __init__(self, taskId: str, lng: float, lat: float, needChangeNum: int, importance: float):
        self.taskId = taskId
        self.position = Position(lng, lat)
        self.needChangeNum = needChangeNum
        self.importance = importance


class MyRoute:
    def __init__(self, dataList, jobLongTime):
        self.dataList = dataList  # [ExchangeBatteryUser, Task, Task,..., Task, BatterySite]
        self.jobLongTime = jobLongTime  # 该job的持续时间
        self.data = None
        self.length = None

    def _create_data(self):
        user = self.dataList[0]
        self.vehicleCapacity = user.vehicleCapacity
        depot = self.dataList[-1]
        for i in range(6):
            self.dataList.append(depot)
        length = len(self.dataList)
        distance_matrix = np.zeros([length, length], dtype='int')
        for i in range(length):
            for j in range(i + 1, length):
                temp = Distance.manhattanDistance(self.dataList[i], self.dataList[j])
                distance_matrix[i][j] = temp
                distance_matrix[j][i] = temp
        distance_matrix[:, 0] = 0

        start_batteryNum = user.batteryNum if user.batteryNum > 0 else user.vehicleCapacity
        demands = [max(user.vehicleCapacity - start_batteryNum, 0)]
        penalty_list = [0]
        for ele in self.dataList[1:]:
            if isinstance(ele, PowerExchangeTask):
                demands.append(ele.needChangeNum)
                penalty_list.append(ele.importance)
        demands.extend([-1, -2, -4, -8, -16, -32, -64])
        penalty_list.extend([0, 0, 0, 0, 0, 0, 0])
        time_matrix = 3.6 * distance_matrix / user.rideSpeed  # 矩阵的每个元素单位是秒
        time_matrix = np.round(time_matrix).astype(int)

        battery_time = np.array(demands) * user.changeBatterySpeed
        battery_time[-7:] = 0

        penalty_list = np.array(penalty_list, dtype='int64') * 2000

        data = {}
        data['vehicle_capacities'] = [user.vehicleCapacity]
        data['num_vehicles'] = 1
        data['depot'] = 0
        data['demands'] = demands
        data['distance_matrix'] = distance_matrix.tolist()
        data['time_matrix'] = time_matrix.tolist()
        data['battery_time'] = battery_time.tolist()
        data['penalty_list'] = penalty_list.tolist()
        self.data = data
        self.length = len(data['demands'])

    def _outputRoute(self, manager, routing, assignment) -> List[PowerExchangeTask]:
        index = routing.Start(0)
        route_index_list = []
        while not routing.IsEnd(index):
            route_index_list.append(manager.IndexToNode(index))
            index = assignment.Value(routing.NextVar(index))
        routes = []
        for i in route_index_list[1:]:
            if isinstance(self.dataList[i], PowerExchangeTask):
                routes.append(self.dataList[i])
        return routes

    def getRoute(self) -> List[PowerExchangeTask]:
        self._create_data()
        manager = pywrapcp.RoutingIndexManager(self.length, self.data['num_vehicles'], self.data['depot'])
        routing = pywrapcp.RoutingModel(manager)

        # 声明距离定义函数
        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return self.data['distance_matrix'][from_node][to_node]

        # 用定义的距离作为cost
        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # 添加约束1-容量约束
        def demand_callback(from_index):
            from_node = manager.IndexToNode(from_index)
            return self.data['demands'][from_node]

        demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
        capacity = 'Capacity'
        routing.AddDimensionWithVehicleCapacity(
            demand_callback_index,
            0,  # null capacity slack
            self.data['vehicle_capacities'],  # vehicle maximum capacities
            True,  # start cumul to zero
            capacity)

        capacity_dimension = routing.GetDimensionOrDie(capacity)
        for location_idx in range(self.length):
            index = manager.NodeToIndex(location_idx)
            capacity_dimension.CumulVar(index).SetRange(0, self.vehicleCapacity)

        # 添加约束2-时间约束
        def time_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return self.data['time_matrix'][from_node][to_node] + self.data['battery_time'][from_node]

        time_callback_index = routing.RegisterTransitCallback(time_callback)
        routing.AddDimension(
            time_callback_index,
            0,  # 没有时间缓冲
            self.jobLongTime * 3600,  # 最多排一小时
            True,  # 以0开始
            'time')

        # Allow to drop nodes.
        for node in range(1, self.length):
            routing.AddDisjunction([manager.NodeToIndex(node)], self.data['penalty_list'][node])

        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
        assignment = routing.SolveWithParameters(search_parameters)
        routes = self._outputRoute(manager, routing, assignment)
        return routes  # 不包括人员的位置信息


def prepareData(received_data):
    dataList = []
    driver = received_data['driver']
    depot = received_data['batterySite']
    tasks = received_data['tasks']

    if driver.get('initFullBatteryNumber', -1) == -1:
        initFullBatteryNumber = driver['vehicleCapacity']
    else:
        initFullBatteryNumber = driver['initFullBatteryNumber']
    exchangeBatteryWorker = ExchangeBatteryUser(initFullBatteryNumber,
                                                driver['driverLocation']['lng'],
                                                driver['driverLocation']['lat'],
                                                driver['vehicleCapacity'])
    dataList.append(exchangeBatteryWorker)
    if driver['vehicleCapacity'] == 0:
        log.info('载具容量为零')

    for task in tasks:
        dataList.append(PowerExchangeTask(
            task['taskId'],
            task['taskLocation']['lng'],
            task['taskLocation']['lat'],
            task['needExchangeNumber'],
            task['score']))

    batterySite = BatterySite(depot['lng'], depot['lat'])
    dataList.append(batterySite)

    return dataList


def getNearestTasks(targetTask: PowerExchangeTask, dataList, distance=200) -> List[PowerExchangeTask]:
    allTasks = dataList[1:-1]
    temp0 = []
    for i, task in enumerate(allTasks):
        dis = Distance.lineDistance(task, targetTask)
        temp0.append([i, dis])
    temp1 = list(filter(lambda x: x[1] < distance, temp0))
    temp2 = sorted(temp1, key=lambda x: x[1])
    if len(temp2) > 5:
        temp3 = temp2[:5]
    else:
        temp3 = temp2
    result = []
    for index, _ in temp3:
        result.append(allTasks[index])
    return result


def outputRoute(dataList, jobDuration=1):
    myRoute = MyRoute(dataList, jobDuration)
    taskList = myRoute.getRoute()
    routes = []
    if len(taskList) == 0:
        log.info("路径规划生成的结果为空！！！")
        return {'code': 0, 'msg': '路径规划生成的结果为空！！！', 'data': None}

    outputResult = getNearestTasks(taskList[0], dataList)
    for task in outputResult:
        routes.append({'taskId': task.taskId})
    output_data = {'code': 0, 'msg': '', 'data': routes}
    return output_data


# --------------------------------------------

class Task:
    def __init__(self, taskId, taskType, canBeSplit, taskLocation, taskDispatchNumber, siteGuid):
        self.taskId = taskId
        self.taskType = taskType
        self.canBeSplit = canBeSplit
        self.position = taskLocation
        self.taskDispatchNumber = taskDispatchNumber
        self.siteGuid = siteGuid


class CSolution(object):
    """docstring for CSolution"""

    def __init__(self, gain, distance, finished, route):
        super(CSolution, self).__init__()
        self.gain = gain
        self.distance = distance
        self.finished = finished
        self.route = route

    def __repr__(self):
        return str({'gain': self.gain, 'distance': self.distance, 'finished': self.finished, 'route': self.route})


class CPark(object):
    """docstring for CPark"""

    def __init__(self, point, supply):
        super(CPark, self).__init__()
        self.point = point
        self.supply = supply
        self.listSolution = []

    def GetNewSolutions(self, supply, distance, capacity):
        listRet = []
        for solution in self.listSolution:
            if solution.finished == True:
                continue
            if solution.gain + supply >= capacity:
                finished = True
                gain = capacity
            else:
                finished = False
                gain = solution.gain + supply
            distance = solution.distance + distance
            listRet.append(CSolution(gain, distance, finished, solution.route))
        return listRet

    def AddSolutions(self, listSolution, k):
        listSolution = copy.deepcopy(listSolution)
        for solution in listSolution:
            solution.route += [self.point]
        self.listSolution += listSolution
        self.listSolution = sorted(self.listSolution,
                                   key=lambda x: (1 if x.finished else 0, x.gain / float(x.distance)), reverse=True)
        if len(self.listSolution) > k:
            self.listSolution = self.listSolution[:k]


class CProblem(object):
    """docstring for CProblem"""

    # @param matrixDistance: distance between every two parks. In meter(m)
    # @param arraySupply: evBike to unload in every park
    # @param capacity: #evBike on truck
    # @param k: beam search parameter
    # @parma minDistance: no park within radius of minDistance of startPoint
    # @param maxDistance: only park within maxDistance considered available
    # @param startPoint: vehicle start point
    def __init__(self, matrixDistance, arraySupply, capacity, k, minDistance, maxDistance, startPoint):
        super(CProblem, self).__init__()
        self.matrixDistance = matrixDistance
        self.arraySupply = arraySupply
        self.capacity = capacity
        self.k = k
        self.minDistance = minDistance
        self.maxDistance = maxDistance
        self.startPoint = startPoint
        self.listPark = [CPark(i, supply) for i, supply in enumerate(self.arraySupply)]
        self.listPark[startPoint].listSolution.append(CSolution(0, 0, False, [0]))

    def CalRoute(self):
        dictVisited = {}
        queue = [self.startPoint]
        # if queue is empty, terminate
        while True:
            # for park in self.listPark:
            #   print park.listSolution
            # print '==='
            if queue == []:
                break
            # dequeue
            curPoint = queue.pop(0)
            # visit every neighbour of this park
            for neighbourPoint, distance in enumerate(self.matrixDistance[curPoint]):
                if curPoint == neighbourPoint:
                    continue
                if distance > self.maxDistance:
                    continue
                # if self.matrixDistance[self.startPoint][curPoint] < self.minDistance:
                if curPoint != self.startPoint and self.matrixDistance[self.startPoint][curPoint] < self.minDistance:
                    continue
                if neighbourPoint in dictVisited:
                    continue
                # get top new solutions
                listSolution = self.listPark[curPoint].GetNewSolutions(self.arraySupply[neighbourPoint], distance,
                                                                       self.capacity)
                # update neighbour information
                self.listPark[neighbourPoint].AddSolutions(listSolution, self.k)
                # add neighbour into queue
                queue.append(neighbourPoint)
            # update visited list
            dictVisited[curPoint] = True
            # sort queue
            queue = list(set(queue))
            queue = sorted(queue, key=lambda x: (
                1 if len(self.listPark[x].listSolution) > 0 and self.listPark[x].listSolution[0].finished else 0,
                self.listPark[x].listSolution[0].gain / float(self.listPark[x].listSolution[0].distance) if len(
                    self.listPark[x].listSolution) > 0 else 0), reverse=True)

        listCandidate = []
        for park in self.listPark:
            for solution in park.listSolution:
                if solution.finished == True:
                    listCandidate.append(solution)
        listCandidate = sorted(listCandidate, key=lambda x: x.gain / float(x.distance), reverse=True)
        if len(listCandidate) == 0:
            return []
        else:
            return listCandidate[0].route


def preProcess(received_data):
    vehicleCapacity = received_data['driver']['vehicleCapacity']
    tasks = []
    for dic in received_data['tasks']:
        taskLocation = Position(float(dic['taskLocation']['lng']), float(dic['taskLocation']['lat']))
        task = Task(dic['taskId'], dic['taskType'], dic['splittable'], taskLocation, dic['taskDispatchNumber'],
                    dic['siteGuid'])
        tasks.append(task)
    length = len(tasks)
    distance_matrix = np.zeros([length, length])
    for i in range(length):
        for j in range(i + 1, length):
            temp = Distance.manhattanDistance(tasks[i], tasks[j])
            distance_matrix[i][j] = temp
            distance_matrix[j][i] = temp
    matrixDistance = distance_matrix.tolist()
    arraySupply = [task.taskDispatchNumber for task in tasks]
    arraySupply[0] = 0
    k = 3
    minDistance = 100
    maxDistance = 3000  # meter
    startPoint = 0

    firstSolution = tasks[0]
    if firstSolution.canBeSplit:
        if firstSolution.taskDispatchNumber <= vehicleCapacity:
            capacity = firstSolution.taskDispatchNumber
        else:
            if firstSolution.taskDispatchNumber <= vehicleCapacity + 2:
                capacity = firstSolution.taskDispatchNumber
            else:
                capacity = vehicleCapacity
    else:
        capacity = firstSolution.taskDispatchNumber
    problem = CProblem(matrixDistance, arraySupply, capacity, k, minDistance, maxDistance, startPoint)
    listRoute = problem.CalRoute()
    routes = []
    if len(listRoute) == 0:
        print("listRoute length is 0")
    elif len(listRoute) == 1:
        print("listRoute length is 1")
        for i in listRoute:
            routes.append({'taskId': tasks[i].taskId, 'consumption': tasks[i].taskDispatchNumber})
    else:
        leftNum = sum(tasks[i].taskDispatchNumber for i in listRoute[1:]) - tasks[0].taskDispatchNumber
        for i in listRoute[:-1]:
            routes.append({'taskId': tasks[i].taskId, 'consumption': tasks[i].taskDispatchNumber})
        routes.append(
            {'taskId': tasks[listRoute[-1]].taskId, 'consumption': tasks[listRoute[-1]].taskDispatchNumber - leftNum})
    if routes:
        output_data = {'code': 0, 'msg': '', 'data': routes}
    else:
        output_data = {'code': 500, 'msg': 'distance is inappropriate', 'data': None}
    return output_data


@app.route("/evbike-schedule/route-plan", methods=["POST"])
def evbike_schedule():
    log.info("助力车调度的路径规划服务")
    try:
        received_data = request.json  # post请求的参数
        output_data = preProcess(received_data)
    except Exception as e:
        log.info("出现异常")
        output_data = {'code': 500, 'msg': e.__doc__, 'data': None}
    return jsonify(output_data)


@app.route("/evbike-power-exchange/route-plan", methods=["POST"])
def power_exchange():
    log.info("智能换电的路径规划服务")
    try:
        received_data = request.json  # post请求的参数
        dataList = prepareData(received_data)
        output_data = outputRoute(dataList)
    except Exception as e:
        output_data = {'code': 500, 'msg': e.__doc__, 'data': None}
        log.info("出现异常")
    return jsonify(output_data)


@app.route("/", methods=["HEAD"])
def my_head():
    output_data = {'code': 200, 'msg': None, 'data': None}
    return jsonify(output_data)


# app.run(host="0.0.0.0", port=22889)
# gunicorn -w 1 -b 0.0.0.0:52289 -D server_route_diaodu:app
