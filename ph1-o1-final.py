import random
import math
import networkx as nx
import matplotlib.pyplot as plt
import os

# --------------------------------------------------------
# تنظیم پارامترهای کلی
# --------------------------------------------------------

NUM_TASKS = 10  # تعداد وظایفی که تولید خواهد شد
NODES_RANGE = (5, 20)  # بازه‌ی تعداد گره‌های میانی هر وظیفه
WCET_RANGE = (13, 30)  # بازه‌ی بدترین زمان اجرای گره‌ها
P_EDGE = 0.1  # احتمال ایجاد یال در روش Erdős–Rényi
D_RATIO_RANGE = (0.125, 0.25)  # نسبت طول مسیر بحرانی به مهلت (جهت تعیین D_i)
RESOURCE_RANGE = (1, 6)  # بازه‌ی تعداد منابع مشترک
ACCESS_COUNT_RANGE = (1, 16)  # بازه‌ی تعداد کل دسترسی به هر منبع
ACCESS_LEN_RANGE = (5, 100)  # بازه‌ی حداکثر طول دسترسی (Critical Section)

HARD_SOFT_PROB = 0.4  # احتمال تبدیل گره‌ی Hard به Soft (در صورت عدم نقض قید)
CRITICAL_RATIOS = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4]  # نسبت گره‌های بحرانی به غیربحرانی
OUTPUT_DIR = "graphs_output"  # پوشه‌ای برای ذخیره تصاویر گراف‌ها

# اگر پوشه‌ی خروجی وجود ندارد، بسازیم
os.makedirs(OUTPUT_DIR, exist_ok=True)


# --------------------------------------------------------
# توابع کمکی جهت ساخت گراف و محاسبه‌ی پارامترها
# --------------------------------------------------------

def create_random_dag(num_inner, p):
    """
    با استفاده از NetworkX، یک DAG تصادفی با num_inner گره میانی (0..num_inner-1) می‌سازیم.
    برای هر جفت (u < v)، با احتمال p یال جهت‌دار (u -> v) افزوده می‌شود (روبه‌جلو).
    در نهایت، یک DiGraph بدون حلقه تولید می‌گردد.
    """
    total_nodes = num_inner + 2
    while True:
        G = nx.DiGraph()
        G.add_nodes_from(range(num_inner))

        # 1) انواع گره‌ها: پیش‌فرض همه‌ی میانی‌ها Hard؛ source/sink = None
        node_types = {}
        for n in range(total_nodes):
            node_types[n] = None
        for n in range(num_inner):
            node_types[n] = "Soft"

        # 3) تعیین گره‌های بحرانی (Critical) براساس نسبت تصادفی
        ratio_crit = random.choice(CRITICAL_RATIOS)  # انتخاب نسبت از لیست
        ratio_crit /= ratio_crit + 1
        num_crit = round(ratio_crit * num_inner)  # تعداد گره‌های بحرانی
        # انتخاب تصادفی این تعداد از میان گره‌های میانی
        crit_nodes = set(random.sample(range(num_inner), num_crit)) if num_inner > 0 else set()

        # 2) تبدیل برخی گره‌ها به Soft با احتمال HARD_SOFT_PROB (بدون نقض قید)
        #    قید: اگر گره n را Soft کنیم، فرزندانش نباید Hard باشند
        for n in crit_nodes:
            node_types[n] = "Hard"

        for u in range(num_inner):
            for v in range(u + 1, num_inner):
                if random.random() < p and not (node_types[u] == "Soft" and node_types[v] == "Hard"):
                    G.add_edge(u, v)
        try:
            list(nx.topological_sort(G))
            return G, node_types, ratio_crit
        except nx.NetworkXUnfeasible:
            continue


def compute_longest_path_length_dag(G, wcet):
    """
    محاسبه‌ی طول مسیر بحرانی (L_i) در گراف جهت‌دار G با استفاده از:
      - wcet[node] = زمان اجرای گره
    الگوریتم: ترتیب توپولوژیک + برنامه‌نویسی پویا
    """
    topo_order = list(nx.topological_sort(G))
    dp = {node: wcet[node] for node in G.nodes()}
    for u in topo_order:
        for v in G.successors(u):
            dp[v] = max(dp[v], dp[u] + wcet[v])
    return max(dp.values()) if dp else 0


def generate_one_task(task_id):
    """
    ساخت یک وظیفه‌ی DAG با گره‌های میانی + source, sink
    مراحل:
      1) تولید گراف میانی با create_random_dag
      2) افزودن گره source (اندیس = num_inner) و sink (اندیس = num_inner+1)
      3) انتساب Hard/Soft به گره‌های میانی (با رعایت قید ساده)
      4) انتخاب نسبت تصادفی گره‌های بحرانی (Critical) و مشخص‌کردن آن‌ها
      5) محاسبه‌ی C_i, L_i, D_i, T_i, U_i
      6) رسم و ذخیره‌ی تصویر گراف
    """
    # تعداد گره‌های میانی
    num_inner = random.randint(*NODES_RANGE)

    # ساخت گراف میانی
    G_mid, node_types, ratio_crit = create_random_dag(num_inner, P_EDGE)

    # گره‌های source و sink
    source_id = num_inner
    sink_id = num_inner + 1
    total_nodes = num_inner + 2  # 0..(num_inner-1), plus source, sink

    # ساخت DiGraph نهایی
    G = nx.DiGraph()
    G.add_nodes_from(range(total_nodes))
    # کپی یال‌های گراف میانی
    for (u, v) in G_mid.edges():
        G.add_edge(u, v)

    # اضافه‌کردن یال حداقلی: source->0 و (num_inner-1)->sink (اگر گره میانی وجود دارد)
    if num_inner > 0:
        G.add_edge(source_id, 0)
        G.add_edge(num_inner - 1, sink_id)

    # 4) انتساب WCET
    wcet = {}
    for n in range(total_nodes):
        if n == source_id or n == sink_id:
            wcet[n] = 0
        else:
            wcet[n] = random.randint(*WCET_RANGE)

    # C_i = مجموع زمان‌های اجرای گره‌های میانی
    Ci = sum(wcet[n] for n in range(num_inner))

    # محاسبه‌ی طول مسیر بحرانی
    Li = compute_longest_path_length_dag(G, wcet)

    # تعیین D_i بر اساس نسبت تصادفی از [0.125..0.25]
    ratio_d = random.uniform(*D_RATIO_RANGE)
    if ratio_d == 0:
        Di = Li
    else:
        Di = int(Li / ratio_d)

    Ti = Di
    Ui = Ci / Ti if Ti > 0 else float('inf')

    # 5) رسم گراف با رنگ‌های متفاوت برای Hard/Soft/Source/Sink
    pos = nx.spring_layout(G, seed=10)  # برای تکرارپذیری چیدمان

    plt.figure(figsize=(8, 6))
    node_colors = []
    for n in G.nodes():
        if n == source_id or n == sink_id:
            node_colors.append("yellow")
        elif node_types[n] == "Hard":
            node_colors.append("red")
        elif node_types[n] == "Soft":
            node_colors.append("green")
        else:
            node_colors.append("gray")

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=500, alpha=0.8)
    nx.draw_networkx_edges(G, pos, edge_color='black', arrows=True)
    # برچسب گره = ID (و شاید نشان دهیم این گره critical هست یا نه)
    labels_dict = {}
    for n in G.nodes():
        labels_dict[n] = f"{n}"
    nx.draw_networkx_labels(G, pos, labels=labels_dict, font_color='white')

    plt.title(f"Task {task_id} - N={num_inner} | ratio_crit={ratio_crit}")
    plt.axis('off')
    output_path = os.path.join(OUTPUT_DIR, f"dag_task_{task_id}.png")
    plt.savefig(output_path, dpi=150)
    plt.close()

    return {
        "task_id": task_id,
        "num_nodes": total_nodes,
        "edges": list(G.edges()),
        "node_types": node_types,
        "critical_ratio": ratio_crit,
        "wcet": wcet,
        "C": Ci,
        "L": Li,
        "D": Di,
        "T": Ti,
        "U": Ui
    }


def generate_resources(num_tasks):
    """
    تولید تعداد منابع مشترک و تعیین ویژگی‌های هر منبع:
      - مقدار تصادفی در RESOURCE_RANGE برای n_r
      - برای هر منبع، تعیین حداکثر طول دسترسی (max_access_len) در بازه ACCESS_LEN_RANGE
      - تعداد کل دسترسی (total_accesses) در بازه ACCESS_COUNT_RANGE
      - توزیع این دسترسی میان وظایف، به‌صورت تصادفی
    """
    n_r = random.randint(*RESOURCE_RANGE)
    resources_info = []
    for r_id in range(n_r):
        max_len = random.randint(*ACCESS_LEN_RANGE)
        total_acc = random.randint(*ACCESS_COUNT_RANGE)

        distribution = [0] * num_tasks
        remain = total_acc
        for i in range(num_tasks - 1):
            if remain <= 0:
                break
            pick = random.randint(0, remain)
            distribution[i] = pick
            remain -= pick
        distribution[-1] += remain

        resources_info.append({
            "resource_id": r_id + 1,
            "max_access_length": max_len,
            "total_accesses": total_acc,
            "distribution": distribution
        })

    return n_r, resources_info


def compute_processors_federated(tasks):
    """
    محاسبه‌ی تعداد پردازنده‌های موردنیاز در روش فدرِیتد (Federated Scheduling).
      اگر U_i > 1 => m_i = ceil((C_i - L_i)/(D_i - L_i))
      اگر U_i <= 1 => یک پردازنده‌ی اختصاصی
    """
    total = 0
    for t in tasks:
        Ui = t["U"]
        Ci = t["C"]
        Li = t["L"]
        Di = t["D"]
        if Ui > 1:
            denom = (Di - Li) if (Di - Li) > 0 else 1
            top = (Ci - Li)
            m_i = math.ceil(top / denom)
            if m_i < 1:
                m_i = 1
            total += m_i
        else:
            total += 1
    return total


# --------------------------------------------------------
# تابع اصلی
# --------------------------------------------------------

def main():
    # برای ثابت‌بودن اعداد تصادفی: random.seed(0)

    # 1) ساخت وظایف
    tasks = []
    for i in range(NUM_TASKS):
        t_info = generate_one_task(i + 1)
        tasks.append(t_info)

    # 2) محاسبه‌ی بهره‌وری کل
    U_sum = sum(t["U"] for t in tasks)

    # 3) ساخت منابع مشترک
    n_r, resources_info = generate_resources(NUM_TASKS)

    # 4) تعداد پردازنده‌ها بر اساس جمع بهره‌وری
    m_simple = math.ceil(U_sum)

    # 5) تعداد پردازنده‌ها به روش Federated
    m_fed = compute_processors_federated(tasks)

    # 6) نمایش نتایج در خروجی
    print("==================================================")
    print(" تولید وظایف و منابع (فاز اول) ")
    print("==================================================")

    for t in tasks:
        print(f"\n--- Task tau_{t['task_id']} ---")
        print(f" • تعداد کل گره‌ها (شامل source و sink): {t['num_nodes']}")
        print(f" • نسبت گره‌های بحرانی (Critical): {t['critical_ratio']}")
        print(" • گره‌ها (ID -> WCET, Type):")
        for n in range(t['num_nodes']):
            wc = t['wcet'][n]
            tp = t['node_types'][n]
            print(f"    - {n}: c={wc}, type={tp}")
        print(f" • یال‌ها: {t['edges']}")
        print(f" • C{t['task_id']} = {t['C']}")
        print(f" • L{t['task_id']} = {t['L']}")
        print(f" • D{t['task_id']} = {t['D']}")
        print(f" • T{t['task_id']} = {t['T']}")
        print(f" • U{t['task_id']} = {t['U']:.3f}")
        png_path = os.path.join(OUTPUT_DIR, f"dag_task_{t['task_id']}.png")
        print(f" -> تصویر گراف در فایل: {png_path}")

    print("--------------------------------------------------")
    print(f" • بهره‌وری کل وظایف (UΣ) = {U_sum:.3f}")
    print("--------------------------------------------------")

    print("\n==================================================")
    print(" منابع اشتراکی ")
    print("==================================================")
    print(f" • تعداد منابع (n_r) = {n_r}")
    for r in resources_info:
        dist_str = ", ".join([f"tau_{idx + 1}={val}" for idx, val in enumerate(r['distribution'])])
        print(f"  - منبع l_{r['resource_id']}:")
        print(f"      حداکثر طول دسترسی = {r['max_access_length']}")
        print(f"      تعداد کل دسترسی = {r['total_accesses']}")
        print(f"      توزیع بین وظایف: {dist_str}")

    print("\n==================================================")
    print(" تعداد پردازنده موردنیاز ")
    print("==================================================")
    print(f" • بر اساس جمع بهره‌وری: m = ceil(UΣ) = {m_simple}")
    print(f" • بر اساس Federated Scheduling: m = {m_fed}")
    print("==================================================")


if __name__ == "__main__":
    main()
