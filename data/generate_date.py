import pandas as pd
import random
from datetime import datetime, timedelta


# ==================== 实体数据生成 ====================

def generate_entities():
    """生成实体数据"""
    entities = {}

    # 1. 驱逐舰实体 (80条)
    destroyer_names = [f"驱逐舰-{i:03d}" for i in range(1, 81)]
    countries = ["中国", "美国", "俄罗斯", "日本", "韩国", "英国", "法国", "印度"]
    statuses = ["在役", "退役", "未服役"]

    destroyers = []
    for name in destroyer_names:
        destroyers.append({
            "实体类型": "武器装备/平台类/海上平台/驱逐舰",
            "实体名称": name,
            "名称": name,
            "情况简介": f"{name}是一型现代化导弹驱逐舰",
            "地位作用": "舰队防空、反舰、反潜作战主力",
            "主要用途": "区域防空、反舰打击、反潜作战",
            "型号": f"DDG-{random.randint(100, 999)}",
            "舷号": f"DDG-{random.randint(100, 999)}",
            "研制厂商": random.choice(["江南造船厂", "大连造船厂", "英格尔斯造船厂", "巴斯钢铁造船厂"]),
            "服役时间": (datetime(1990, 1, 1) + timedelta(days=random.randint(0, 10000))).strftime("%Y-%m-%d"),
            "服役状态": random.choice(statuses),
            "母港": random.choice(["青岛", "舟山", "湛江", "诺福克", "圣地亚哥", "横须贺"]),
            "所属国家": random.choice(countries),
            "舷长(米)": round(random.uniform(130, 180), 1),
            "舷宽(米)": round(random.uniform(15, 22), 1),
            "满载排水量(吨)": round(random.uniform(6000, 13000), 0),
            "标准排水量(吨)": round(random.uniform(5000, 10000), 0),
            "吃水(米)": round(random.uniform(5, 9), 1),
            "续航(海里)": round(random.uniform(4000, 8000), 0),
            "最高航速(节)": round(random.uniform(28, 35), 1),
            "自持力(天)": random.randint(30, 90),
            "动力系统": random.choice(["燃气轮机", "柴燃联合动力", "全燃联合动力"])
        })
    entities["武器装备-平台类-海上平台-驱逐舰"] = destroyers

    # 2. 航空母舰实体 (50条)
    carrier_names = [f"航空母舰-{i:03d}" for i in range(1, 51)]
    carriers = []
    for name in carrier_names:
        carriers.append({
            "实体类型": "武器装备/平台类/海上平台/航空母舰",
            "实体名称": name,
            "名称": name,
            "情况简介": f"{name}是一型大型航空母舰",
            "地位作用": "海上移动机场，战略威慑核心",
            "主要用途": "制空、制海、对地攻击",
            "型号": f"CVN-{random.randint(60, 99)}",
            "舷号": f"CVN-{random.randint(60, 99)}",
            "研制厂商": random.choice(["纽波特纽斯造船厂", "江南造船厂", "大连造船厂"]),
            "服役时间": (datetime(1980, 1, 1) + timedelta(days=random.randint(0, 15000))).strftime("%Y-%m-%d"),
            "服役状态": random.choice(statuses),
            "所属国家": random.choice(["中国", "美国", "英国", "法国", "印度", "俄罗斯"]),
            "母港": random.choice(["青岛", "三亚", "诺福克", "圣地亚哥", "朴茨茅斯"]),
            "舷长(米)": round(random.uniform(280, 340), 1),
            "舷宽(米)": round(random.uniform(35, 45), 1),
            "标准排水量(吨)": round(random.uniform(50000, 100000), 0),
            "满载排水量(吨)": round(random.uniform(60000, 120000), 0),
            "吃水(米)": round(random.uniform(10, 13), 1),
            "飞行甲板宽度(米)": round(random.uniform(60, 80), 1),
            "舰员编制": f"{random.randint(3000, 5500)}人",
            "动力系统": random.choice(["核动力", "常规动力"]),
            "航速(节)": round(random.uniform(28, 35), 1),
            "自持力(天)": random.randint(60, 180),
            "机库面积(平方米)": round(random.uniform(3000, 6000), 0),
            "弹射器数量": random.randint(2, 4),
            "弹射方式": random.choice(["蒸汽弹射", "电磁弹射"]),
            "舰载机容量": f"{random.randint(40, 90)}架"
        })
    entities["武器装备-平台类-海上平台-航空母舰"] = carriers

    # 3. 军兵种部队实体 (60条)
    unit_names = [f"部队-{i:03d}" for i in range(1, 61)]
    units = []
    for name in unit_names:
        units.append({
            "实体类型": "作战力量/军兵种部队",
            "实体名称": name,
            "名称": name,
            "部队类型": random.choice(["海军舰队", "空军联队", "陆军师", "海军陆战队", "战略火箭军"]),
            "情况简介": f"{name}是一支精锐作战部队",
            "部队番号": f"第{random.randint(1, 999)}部队",
            "职能任务": random.choice(["远洋作战", "区域防空", "两栖登陆", "战略威慑"]),
            "部署情况": random.choice(["太平洋", "大西洋", "印度洋", "地中海"]),
            "下辖力量": f"{random.randint(3, 15)}个下属单位",
            "隶属部队": random.choice(["海军司令部", "空军司令部", "战区司令部"]),
            "主战装备": random.choice(["驱逐舰", "航空母舰", "战斗机", "导弹系统"]),
            "主要领导": f"指挥官-{random.randint(1, 100)}"
        })
    entities["作战力量-军兵种部队"] = units

    # 4. 海军基地实体 (40条)
    base_names = [f"海军基地-{i:03d}" for i in range(1, 41)]
    bases = []
    for name in base_names:
        bases.append({
            "实体类型": "作战目标/军事目标/军事基地/海军基地",
            "实体名称": name,
            "名称": name,
            "情况简介": f"{name}是重要的海军驻泊基地",
            "地位作用": "舰艇驻泊、维修、补给中心",
            "所属国家": random.choice(countries),
            "启用日期": (datetime(1950, 1, 1) + timedelta(days=random.randint(0, 25000))).strftime("%Y-%m-%d"),
            "服役状态": random.choice(["在役", "退役", "未服役"]),
            "发展历程": "历经多次扩建改造",
            "地理位置": f"东经{random.uniform(100, 180):.2f}, 北纬{random.uniform(10, 60):.2f}",
            "占地面积(平方千米)": round(random.uniform(5, 50), 1),
            "周边环境": random.choice(["沿海城市", "偏远海岛", "河口港湾"]),
            "码头": f"{random.randint(5, 20)}个泊位",
            "船坞": f"{random.randint(1, 5)}座干船坞",
            "维修设施": "具备大修能力",
            "防御情况": "配备岸防导弹和高炮系统"
        })
    entities["作战目标-军事目标-军事基地-海军基地"] = bases

    # 5. 军事人物实体 (50条)
    person_names = [f"将领-{i:03d}" for i in range(1, 51)]
    persons = []
    for name in person_names:
        persons.append({
            "实体类型": "军政人物/军事人物",
            "实体名称": name,
            "中文名": name,
            "英文名": f"General-{random.randint(1000, 9999)}",
            "出生年月": (datetime(1950, 1, 1) + timedelta(days=random.randint(0, 20000))).strftime("%Y-%m-%d"),
            "国籍": random.choice(countries),
            "性别": random.choice(["男", "女"]),
            "情况简介": f"{name}是一位资深军事指挥官",
            "现职": random.choice(["舰队司令", "战区司令", "参谋长", "国防部长"]),
            "军种": random.choice(["海军", "空军", "陆军", "海军陆战队"]),
            "军衔": random.choice(["上将", "中将", "少将", "准将"]),
            "任职经历": "历任多个指挥岗位",
            "教育情况": random.choice(["军事学院", "国防大学", "海军军官学校"]),
            "学历": random.choice(["本科", "硕士", "博士"]),
            "决策风格": random.choice(["果断型", "稳健型", "创新型"]),
            "所属党派": random.choice(["无党派", "执政党"])
        })
    entities["军政人物-军事人物"] = persons

    # ... 可以继续添加更多实体类型 ...

    return entities


# ==================== 关系数据生成 ====================

def generate_relations(entities):
    """生成关系数据"""
    relations = []

    # 获取各类实体列表
    destroyers = entities.get("武器装备-平台类-海上平台-驱逐舰", [])
    carriers = entities.get("武器装备-平台类-海上平台-航空母舰", [])
    units = entities.get("作战力量-军兵种部队", [])
    bases = entities.get("作战目标-军事目标-军事基地-海军基地", [])
    persons = entities.get("军政人物-军事人物", [])

    # 1. 常驻关系 (300条)
    all_ships = destroyers + carriers
    for _ in range(300):
        ship = random.choice(all_ships)
        base = random.choice(bases)
        relations.append({
            "关系名称": "常驻",
            "源实体类型": ship["实体类型"],
            "源实体": ship["实体名称"],
            "目标实体类型": base["实体类型"],
            "目标实体": base["实体名称"]
        })

    # 2. 服役部队关系 (350条)
    for _ in range(350):
        ship = random.choice(all_ships)
        unit = random.choice(units)
        relations.append({
            "关系名称": "服役部队",
            "源实体类型": ship["实体类型"],
            "源实体": ship["实体名称"],
            "目标实体类型": unit["实体类型"],
            "目标实体": unit["实体名称"]
        })

    # 3. 服役舰长关系 (100条)
    for _ in range(100):
        ship = random.choice(all_ships)
        person = random.choice(persons)
        relations.append({
            "关系名称": "服役舰长",
            "源实体类型": ship["实体类型"],
            "源实体": ship["实体名称"],
            "目标实体类型": person["实体类型"],
            "目标实体": person["实体名称"]
        })

    # 4. 型号系列关系 (200条)
    for _ in range(200):
        if len(destroyers) >= 2:
            src = random.choice(destroyers)
            tgt = random.choice([d for d in destroyers if d != src])
            relations.append({
                "关系名称": "型号系列",
                "源实体类型": src["实体类型"],
                "源实体": src["实体名称"],
                "目标实体类型": tgt["实体类型"],
                "目标实体": tgt["实体名称"]
            })

    # 5. 隶属关系 (100条)
    for _ in range(100):
        if len(units) >= 2:
            src = random.choice(units)
            tgt = random.choice([u for u in units if u != src])
            relations.append({
                "关系名称": "隶属",
                "源实体类型": src["实体类型"],
                "源实体": src["实体名称"],
                "目标实体类型": tgt["实体类型"],
                "目标实体": tgt["实体名称"]
            })

    # 6. 装备关系 (100条)
    for _ in range(100):
        unit = random.choice(units)
        ship = random.choice(all_ships)
        relations.append({
            "关系名称": "装备",
            "源实体类型": unit["实体类型"],
            "源实体": unit["实体名称"],
            "目标实体类型": ship["实体类型"],
            "目标实体": ship["实体名称"]
        })

    # ... 可以继续添加更多关系类型 ...

    return relations


# ==================== 导出Excel ====================

def export_to_excel(entities, relations):
    """导出数据到Excel文件"""
    # 导出实体数据（每个实体类型一个sheet）
    with pd.ExcelWriter("生成的实体数据.xlsx") as writer:
        for sheet_name, data in entities.items():
            df = pd.DataFrame(data)
            df.to_excel(writer, sheet_name=sheet_name[:31], index=False)  # sheet名最长31字符

    # 导出关系数据
    relations_df = pd.DataFrame(relations)
    relations_df.to_excel("生成的关系数据.xlsx", index=False)

    print(f"实体数据已保存，共{sum(len(v) for v in entities.values())}条")
    print(f"关系数据已保存，共{len(relations)}条")


# 执行生成
if __name__ == "__main__":
    entities = generate_entities()
    relations = generate_relations(entities)
    export_to_excel(entities, relations)