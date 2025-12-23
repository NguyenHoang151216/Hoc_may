from sqlalchemy import Column, Integer, String
from .database import Base

class VideoPolygon(Base):
    __tablename__ = "VideoPolygon"

    id = Column(Integer, primary_key=True, index=True)
    video_name = Column(String(255), unique=True, nullable=False)
    polygon = Column(String(255), nullable=False)

    def __repr__(self):
        return f"<VideoPolygon(video_name='{self.video_name}')>"