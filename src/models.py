from sqlalchemy import Column, String, Integer
from sqlalchemy.ext.declarative import declarative_base
from pgvector.sqlalchemy import Vector


Base = declarative_base()


class Document(Base):
    __tablename__ = 'documents'
    
    id = Column(Integer, primary_key=True)
    text = Column(String)
    url = Column(String)
    embedding = Column(Vector(1536))  # Dimension for text-embedding-3-small
    doc_type = Column(String)  # 'title', 'content', or 'link'