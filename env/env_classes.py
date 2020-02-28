from env.utils import *
from env.utils import compute_graph


class CholeskyTaskGraph():

    def __init__(self, n, p, window):
        self.time = 0
        self.num_steps = 0
        self.n = n
        self.p = p
        self.window = window
        self.task_graph = compute_graph()
        self.task_data = self.task_graph.copy()
        self.task_to_asap = {v: k for (k, v) in enumerate(self.task_data.task_list)}
        self.cluster = Cluster(node_types=np.zeros(p), communication_cost=np.zeros((p, p)))
        self.running = -1 * np.ones(p)  # array of task number
        self.running_task2proc = {}
        self.ProcReady = np.zeros(p)  # for each processor, the time where it becomes available
        self.ReadyTasks = []
        self.processed = {}
        self.current_proc = 0
        self.is_homogene = True

    def reset(self, n):
        self.task_data = self.task_graph.copy()
        self.time = 0
        self.num_steps = 0
        self.running = -1 * np.ones(self.p)
        self.running_task2proc = {}
        self.ProcReady = [0] * self.p
        self.ReadyTasks.append(0)
        self.current_proc = 0
        return self._compute_state()

    def step(self, action):
        """
        first implementation, with only [-1, 0, ..., T] actions
        :param action: -1: does nothing. t: schedules t on the current available processor
        :return: next_state, reward, done, info
        """
        self.num_steps += 1

        self._find_available_proc()

        if action == -1:
            if len(self.running_task2proc) == 0:
                # the agent does nothing but every proc is available: we enforce an arbitrary action
                action = self.ReadyTasks[0]

        self._choose_task_processor(action, self.current_proc)
        done = self._go_to_next_action()
        reward = -self.time if done else 0
        info = {'episode': {'r': reward, 'length': self.num_steps}, 'bad_transition': False}

        return self._compute_state(), reward, done, info

    def _find_available_proc(self):
        while (self.current_proc < self.p) and (self.ProcReady[self.current_proc] > -1):
            self.current_proc += 1
        if self.current_proc == self.p:
            # no new proc available
            self.current_proc == 0
            self.forward_in_time()
        while (self.current_proc < self.p) and (self.ProcReady[self.current_proc] > -1):
            self.current_proc += 1

    def _forward_in_time(self):
        min_time = np.min(self.ProcReady[self.ProcReady > self.time])
        self.time += min_time

        self.ProcReady[self.ProcReady < self.time] = self.time

        tasks_finished = self.running[self.ProcReady == self.time][self.running > -1].copy()
        self.running[tasks_finished] == -1
        for task in tasks_finished:
            del self.running_task2proc[tasks_finished]

        # remove nodes
        self.task_data.remove_nodes(tasks_finished)

        # compute new ready tasks
        mask = isin(self.task_data.edge_index[0], tasks_finished)
        list_succ = self.task_data.edge_index[1][mask]
        list_succ = torch.unique(list_succ)
        new_ready_tasks = list_succ[torch.logical_not(isin(list_succ, self.task_data.edge_index[:, 1]))]
        self.ReadyTasks += new_ready_tasks
        self.current_proc = np.argmin(self.running)

    def _go_to_next_action(self, previous_action):
        has_just_passed = self.is_homogene and previous_action == -1
        if has_just_passed:
            self._forward_in_time()
        while len(self.ReadyTasks) == 0:
            self._forward_in_time()
            if self._isdone():
                return True
        self._find_available_proc()
        return False

    def _choose_task_processor(self, action, processor):
        # assert action in self.ReadyTasks

        if action != -1:
            self.ProcReady[processor] += self.task_data.task_list[action].durations[self.cluster.node_types[processor]]
            self.ReadyTasks.remove(action)
            self.processed[self.task_data.task_list[action].barcode] = [processor, self.time]
            self.running_task2proc[action] = processor

    def _compute_state(self):
        visible_graph = compute_sub_graph(self.task_data,
                                          np.concatenate((self.running[self.running > -1], self.ReadyTasks)),
                                          self.window)
        visible_graph.x = self._compute_embeddings(visible_graph.x)
        return visible_graph

    def _remaining_time(self, running_tasks):
        return torch.tensor([self.ProcReady[self.running_task2proc[task]] for task in running_tasks]) - self.time

    def _isdone(self):
        return (len(self.ReadyTasks) == 0) and (len(self.running_task2proc) == 0)
    
    def _compute_embeddings(self, tasks):

        ready = isin(tasks, torch.tensor(self.ReadyTasks)).unsqueeze(-1).float()
        running = isin(tasks, torch.tensor(self.running[self.running > -1])).unsqueeze(-1).float()

        remaining_time = torch.zeros(tasks.shape[0])
        remaining_time[running] = self._remaining_time(tasks[running])
        remaining_time = remaining_time.unsqueeze(-1)

        n_succ = torch.sum((tasks.unsqueeze(-1) == self.task_data.edge_index[0].unsqueeze(0)).float(), dim=1)
        n_pred = torch.sum((tasks.unsqueeze(-1) == self.task_data.edge_index[1].unsqueeze(0)).float(), dim=1)

        task_type = torch.tensor([task.type for task in tasks]).unsqueeze(-1)
        num_classes = 5
        one_hot_type = (task_type == torch.arange(num_classes).reshape(1, num_classes)).float()

        # add other embeddings

        return torch.cat((n_succ, n_pred, one_hot_type, ready, running, remaining_time), dim=1)

