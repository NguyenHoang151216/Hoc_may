from sqlalchemy.orm import Session
from typing import Optional, List, Tuple
from .models import VideoPolygon

Point = Tuple[int, int]

# =========================
# CONVERT UTILS
# =========================
def polygon_to_string(polygon: List[Point]) -> str:
    """
    [(x1,y1),(x2,y2),(x3,y3),(x4,y4)]
    → "x1,y1,x2,y2,x3,y3,x4,y4"
    """
    flat = []
    for x, y in polygon:
        flat.extend([str(x), str(y)])
    return ",".join(flat)


def string_to_polygon(data: str) -> List[Point]:
    """
    "x1,y1,x2,y2,x3,y3,x4,y4"
    → [(x1,y1),(x2,y2),(x3,y3),(x4,y4)]
    """
    nums = list(map(int, data.split(",")))
    return [(nums[i], nums[i + 1]) for i in range(0, len(nums), 2)]


# =========================
# REPOSITORY
# =========================
class PolygonRepository:
    def __init__(self):
        from .database import SessionLocal
        self.db = SessionLocal()

    def get_polygon_by_video(self, video_name: str) -> Optional[List[Point]]:
        """
        Lấy polygon theo tên video
        Returns: List[(x,y)] hoặc None
        """
        record = (
            self.db.query(VideoPolygon)
            .filter(VideoPolygon.video_name == video_name)
            .first()
        )
        
        if not record:
            return None
        
        return string_to_polygon(record.polygon)

    def save_polygon(self, video_name: str, polygon: List[Point]):
        """
        Lưu polygon (INSERT hoặc UPDATE)
        """
        if not polygon or len(polygon) < 3:
            raise ValueError("Polygon phải có ít nhất 3 điểm")
        
        polygon_str = polygon_to_string(polygon)
        
        # Check existing
        existing = (
            self.db.query(VideoPolygon)
            .filter(VideoPolygon.video_name == video_name)
            .first()
        )
        
        if existing:
            # UPDATE
            existing.polygon = polygon_str
            print(f"[DB] Updated polygon for: {video_name}")
        else:
            # INSERT
            record = VideoPolygon(
                video_name=video_name,
                polygon=polygon_str
            )
            self.db.add(record)
            print(f"[DB] Inserted polygon for: {video_name}")
        
        self.db.commit()

    def delete_polygon(self, video_name: str) -> bool:
        """
        Xóa polygon theo video name
        """
        record = (
            self.db.query(VideoPolygon)
            .filter(VideoPolygon.video_name == video_name)
            .first()
        )
        
        if record:
            self.db.delete(record)
            self.db.commit()
            return True
        
        return False

    def list_all_videos(self) -> List[str]:
        """
        Danh sách video đã có polygon
        """
        records = self.db.query(VideoPolygon.video_name).all()
        return [r.video_name for r in records]

    def close(self):
        """
        Đóng session
        """
        self.db.close()