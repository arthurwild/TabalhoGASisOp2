"""
Arquitetura: 1 Master + 3 Workers
Métricas de escalonamento:
  1. CPU disponível
  2. Memória disponível
  3. Espaço em disco disponível
  4. Latência de rede (ms)
"""

import threading
import queue
import time
import random
import json
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime

@dataclass
class Pod:
    name: str
    cpu_req: float
    mem_req: float
    disk_req: float
    priority: int        # 1 (baixa) até 5 (alta)
    status: str = "PENDING"
    assigned_worker: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    scheduled_at: Optional[float] = None

    def __repr__(self):
        return (f"Pod({self.name} | CPU:{self.cpu_req} MEM:{self.mem_req}GB "
                f"DISK:{self.disk_req}GB PRI:{self.priority} → {self.status})")


@dataclass
class WorkerNode:
    name: str
    total_cpu: float
    total_mem: float
    total_disk: float
    network_latency: float  #latência até o master

    used_cpu: float = 0.0
    used_mem: float = 0.0
    used_disk: float = 0.0
    pods: list = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def free_cpu(self):
        return self.total_cpu - self.used_cpu

    @property
    def free_mem(self):
        return self.total_mem - self.used_mem

    @property
    def free_disk(self):
        return self.total_disk - self.used_disk

    @property
    def cpu_pct(self):
        return (self.used_cpu / self.total_cpu) * 100

    @property
    def mem_pct(self):
        return (self.used_mem / self.total_mem) * 100

    @property
    def disk_pct(self):
        return (self.used_disk / self.total_disk) * 100

    def can_fit(self, pod: Pod) -> bool:
        return (self.free_cpu >= pod.cpu_req and
                self.free_mem >= pod.mem_req and
                self.free_disk >= pod.disk_req)

    def assign(self, pod: Pod):
        with self.lock:
            self.used_cpu  += pod.cpu_req
            self.used_mem  += pod.mem_req
            self.used_disk += pod.disk_req
            self.pods.append(pod.name)

    def release(self, pod: Pod):
        with self.lock:
            self.used_cpu  -= pod.cpu_req
            self.used_mem  -= pod.mem_req
            self.used_disk -= pod.disk_req
            if pod.name in self.pods:
                self.pods.remove(pod.name)

#Algoritmo de escalonamento
def score_worker(worker: WorkerNode, pod: Pod) -> float:
    """
    Pontuação composta (maior = melhor candidato).

    Métrica 1 – CPU livre normalizada       (peso 35%)
    Métrica 2 – Memória livre normalizada   (peso 30%)
    Métrica 3 – Disco livre normalizado     (peso 20%) 
    Métrica 4 – Latência de rede invertida  (peso 15%)

    Cada componente varia de 0 a 1.
    """
    if not worker.can_fit(pod):
        return -1.0  # inelegível

    w_cpu  = 0.35
    w_mem  = 0.30
    w_disk = 0.20
    w_lat  = 0.15

    s_cpu  = worker.free_cpu  / worker.total_cpu
    s_mem  = worker.free_mem  / worker.total_mem
    s_disk = worker.free_disk / worker.total_disk
    # latência: normaliza invertida (máx assumido = 200 ms)
    s_lat  = 1.0 - min(worker.network_latency / 200.0, 1.0)

    return (w_cpu * s_cpu +
            w_mem * s_mem +
            w_disk * s_disk +
            w_lat * s_lat)


class MasterNode:
    def __init__(self, workers: list[WorkerNode]):
        self.workers = workers
        self.pod_queue: queue.PriorityQueue = queue.PriorityQueue()
        self.scheduled_pods: list[Pod] = []
        self.failed_pods: list[Pod] = []
        self.lock = threading.Lock()
        self._running = True
        self._scheduler_thread = threading.Thread(
            target=self._scheduler_loop, name="master-scheduler", daemon=True
        )

    def start(self):
        self._scheduler_thread.start()
        print(f"[{_ts()}] [MASTER] Scheduler iniciado com {len(self.workers)} workers.")

    def stop(self):
        self._running = False

    def submit_pod(self, pod: Pod):
        #Produtor: enfileira o pod (prioridade negativa, maior prioridade sai primeiro).
        self.pod_queue.put((-pod.priority, time.time(), pod))

    def _scheduler_loop(self):
        """Consumidor: retira pods da fila e os escalona."""
        while self._running:
            try:
                _, _, pod = self.pod_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            self._schedule(pod)
            self.pod_queue.task_done()

    def _schedule(self, pod: Pod):
        scores = [(score_worker(w, pod), w) for w in self.workers]
        scores.sort(key=lambda x: x[0], reverse=True)

        best_score, best_worker = scores[0]

        if best_score < 0:
            pod.status = "FAILED"
            with self.lock:
                self.failed_pods.append(pod)
            print(f"[{_ts()}] MASTER: Falha no {pod.name}, sem worker elegível devido a recursos insuficientes")
            return

        best_worker.assign(pod)
        pod.status = "RUNNING"
        pod.assigned_worker = best_worker.name
        pod.scheduled_at = time.time()

        with self.lock:
            self.scheduled_pods.append(pod)

        latency_sim = best_worker.network_latency
        print(
            f"[{_ts()}] MASTER: {pod.name:20s} rodando, melhor worker {best_worker.name:10s} "
            f"(score={best_score:.3f} | CPU:{best_worker.cpu_pct:5.1f}% "
            f"MEM:{best_worker.mem_pct:5.1f}% DISK:{best_worker.disk_pct:5.1f}% "
            f"LAT:{latency_sim:.0f}ms)"
        )

    def status_report(self):

        print("\n\n  RELATÓRIO FINAL\n")
        # Workers
        for w in self.workers:
            print(f"\n- {w.name}")
            print(f"│  Latência: {w.network_latency:.0f} ms")
            bar_cpu  = _bar(w.cpu_pct)
            bar_mem  = _bar(w.mem_pct)
            bar_disk = _bar(w.disk_pct)
            print(f"│  CPU  [{bar_cpu}] {w.used_cpu:.1f}/{w.total_cpu:.1f} núcleos ({w.cpu_pct:.1f}%)")
            print(f"│  MEM  [{bar_mem}] {w.used_mem:.1f}/{w.total_mem:.1f} GB     ({w.mem_pct:.1f}%)")
            print(f"│  DISK [{bar_disk}] {w.used_disk:.1f}/{w.total_disk:.1f} GB     ({w.disk_pct:.1f}%)")
            if w.pods:
                pods_str = ", ".join(w.pods)
                print(f"│  PODs ({len(w.pods)}): {pods_str}")
            else:
                print(f"│  PODs: nenhum")
            print(f"{'-'*55}")

        # Pods agendados
        print(f"\n  PODs AGENDADOS ({len(self.scheduled_pods)}):")
        print(f"  {'Nome':<20} {'Worker':<12} {'CPU':>5} {'MEM':>6} {'DISK':>6} {'PRI':>4}")
        for p in sorted(self.scheduled_pods, key=lambda x: x.assigned_worker):
            print(f"  {p.name:<20} {p.assigned_worker:<12} "
                  f"{p.cpu_req:>5.1f} {p.mem_req:>5.1f}G {p.disk_req:>5.1f}G {p.priority:>4}")

        if self.failed_pods:
            print(f"\n  PODs COM FALHA ({len(self.failed_pods)}):")
            for p in self.failed_pods:
                print(f"    ✗ {p.name} (CPU:{p.cpu_req} MEM:{p.mem_req}G DISK:{p.disk_req}G)")

        total = len(self.scheduled_pods) + len(self.failed_pods)
        print(f"\n  Total submetidos: {total} | Agendados: {len(self.scheduled_pods)} | Falhos: {len(self.failed_pods)}")

# PRODUTOR DE PODS (thread separada)
def pod_producer(master: MasterNode, pods: list[Pod], interval: float = 0.15):
    #Simula chegada de pods em intervalos, como um controlador de deployment.
    for pod in pods:
        master.submit_pod(pod)
        print(f"[{_ts()}] [PRODUCER] Submetido {pod.name} (cpu={pod.cpu_req} mem={pod.mem_req}G pri={pod.priority})")
        time.sleep(interval + random.uniform(0, 0.1))


# HELPERS
def _ts():
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


def _bar(pct: float, width: int = 20) -> str:
    filled = int(pct / 100 * width)
    return "█" * filled + "░" * (width - filled)



# DEFINIÇÃO DO CLUSTER E PODS
def build_cluster() -> list[WorkerNode]:
    return [
        WorkerNode(
            name="worker-1",
            total_cpu=8.0,   total_mem=16.0,  total_disk=100.0,
            network_latency=5.0    # ms - muito próximo do master
        ),
        WorkerNode(
            name="worker-2",
            total_cpu=16.0,  total_mem=32.0,  total_disk=200.0,
            network_latency=12.0   # ms - rack diferente
        ),
        WorkerNode(
            name="worker-3",
            total_cpu=4.0,   total_mem=8.0,   total_disk=50.0,
            network_latency=35.0   # ms - zona diferente / nuvem híbrida
        ),
    ]


def build_pods() -> list[Pod]:
    #20 pods com perfis variados.
    specs = [
        # (nome, cpu, mem_GB, disk_GB, prioridade)
        ("nginx-frontend-1",    0.5,  0.5,  2.0,  3),
        ("nginx-frontend-2",    0.5,  0.5,  2.0,  3),
        ("api-gateway",         1.0,  2.0,  5.0,  5),
        ("auth-service",        0.5,  1.0,  3.0,  5),
        ("user-db-primary",     2.0,  4.0, 20.0,  5),
        ("user-db-replica",     1.5,  3.0, 20.0,  4),
        ("cache-redis",         1.0,  4.0,  1.0,  4),
        ("message-broker",      1.0,  2.0,  10.0, 3),
        ("order-service",       0.5,  1.0,  3.0,  3),
        ("payment-service",     0.5,  1.5,  2.0,  5),
        ("notification-svc",    0.25, 0.5,  1.0,  2),
        ("report-generator",    2.0,  2.0,  15.0, 2),
        ("ml-inference",        4.0,  8.0,  10.0, 3),
        ("batch-job-1",         1.0,  1.0,  5.0,  1),
        ("batch-job-2",         1.0,  1.0,  5.0,  1),
        ("log-aggregator",      0.5,  1.0,  30.0, 2),
        ("metrics-collector",   0.25, 0.5,  5.0,  2),
        ("health-checker",      0.1,  0.25, 1.0,  4),
        ("config-server",       0.25, 0.5,  2.0,  4),
        ("heavy-analytics",     3.0,  6.0,  25.0, 2),
    ]
    return [Pod(name=n, cpu_req=c, mem_req=m, disk_req=d, priority=p)
            for n, c, m, d, p in specs]



# MAIN
def main():
    print("  SIMULADOR DE ESCALONAMENTO KUBERNETES")
    print("  Paradigma: Produtor/Consumidor + Multithreading")
    print("  Métricas: CPU · Memória · Disco · Latência de Rede")

    workers = build_cluster()
    master  = MasterNode(workers)
    pods    = build_pods()

    # Embaralha para simular chegada não-determinística, mas prioridade decide ordem
    random.shuffle(pods)

    # Inicia o master (thread consumidora)
    master.start()
    time.sleep(0.1)

    # Inicia produtor em thread separada
    producer_thread = threading.Thread(
        target=pod_producer, args=(master, pods, 0.12), name="pod-producer"
    )
    producer_thread.start()
    producer_thread.join()

    # Aguarda a fila ser drenada
    master.pod_queue.join()
    time.sleep(0.3)
    master.stop()

    # Exibe relatório final
    master.status_report()


if __name__ == "__main__":
    main()
