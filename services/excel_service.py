import pandas as pd
from datetime import datetime
import json

class ExcelScheduleService:
    def __init__(self, excel_path: str = "schedule.xlsx", mapping_path: str = "mapping.json"):
        self.excel_path = excel_path
        self.mapping_path = mapping_path
        # Пример маппинга по умолчанию, если файла mapping.json еще нет
        self.mapping = {
            "feeding": ["bowl", "cat"],
            "cleaning": ["person", "broom"],
            "sleeping_check": ["cat"]
        }
        # Раскомментируй, когда напарница сделает mapping.json:
        # with open(mapping_path, "r", encoding="utf-8") as f:
        #     self.mapping = json.load(f)

    def get_task_and_targets(self, target_time: datetime):
        """
        Ищет задачу в Excel, куда попадает target_time.
        Предположим, в Excel колонки: 'start_time', 'end_time', 'task_name'
        """
        try:
            # Для заглушки пока возвращаем синтетическую задачу:
            # df = pd.read_excel(self.excel_path)
            # row = df[(df['start_time'] <= target_time) & (df['end_time'] >= target_time)]
            
            # МОК: имитируем, что нашли задачу
            task_name = "feeding" 
            targets = self.mapping.get(task_name, [])
            return task_name, targets
        except Exception as e:
            print(f"Ошибка чтения Excel: {e}")
            return "unknown", []

excel_service = ExcelScheduleService()