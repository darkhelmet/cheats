# SQLAlchemy Cheat Sheet

A comprehensive reference for SQLAlchemy 2.0+ ORM and Core functionality, covering models, sessions, queries, and migrations.

## Table of Contents
- [Quick Start](#quick-start)
- [Basic Setup](#basic-setup)
- [Model Definition](#model-definition)
- [Column Types & Constraints](#column-types--constraints)
- [Relationships](#relationships)
- [Session Management](#session-management)
- [CRUD Operations](#crud-operations)
- [Query Patterns](#query-patterns)
- [Joins & Eager Loading](#joins--eager-loading)
- [Advanced Querying](#advanced-querying)
- [Migrations with Alembic](#migrations-with-alembic)
- [Best Practices](#best-practices)

---

## Quick Start

```python
# Installation
pip install sqlalchemy[asyncio] alembic

# Basic setup
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

class Base(DeclarativeBase):
    pass

# Create engine and session
engine = create_engine("sqlite:///example.db", echo=True)
SessionLocal = sessionmaker(bind=engine)

# Create tables
Base.metadata.create_all(engine)

# Use session
with SessionLocal() as session:
    # Your database operations here
    session.commit()
```

---

## Basic Setup

### Database Connection

```python
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase

# SQLite
engine = create_engine("sqlite:///database.db", echo=True)

# PostgreSQL
engine = create_engine("postgresql://user:password@localhost/dbname")

# MySQL
engine = create_engine("mysql+pymysql://user:password@localhost/dbname")

# Connection with pool settings
engine = create_engine(
    "postgresql://user:pass@localhost/db",
    pool_size=20,
    max_overflow=0,
    pool_pre_ping=True,
    echo=False
)
```

### Declarative Base

```python
# Modern SQLAlchemy 2.0+ approach
from sqlalchemy.orm import DeclarativeBase

class Base(DeclarativeBase):
    pass

# Alternative with type annotations
from typing import Any
from sqlalchemy import MetaData
from sqlalchemy.orm import DeclarativeBase

class Base(DeclarativeBase):
    metadata: MetaData
    type_annotation_map = {
        str: String(50)  # Default string length
    }
```

---

## Model Definition

### Basic Model

```python
from typing import Optional
from sqlalchemy import String, Integer, DateTime, func
from sqlalchemy.orm import Mapped, mapped_column

class User(Base):
    __tablename__ = "users"

    # Primary key
    id: Mapped[int] = mapped_column(primary_key=True)

    # Required fields
    username: Mapped[str] = mapped_column(String(50), unique=True)
    email: Mapped[str] = mapped_column(String(100))

    # Optional fields
    full_name: Mapped[Optional[str]] = mapped_column(String(100))

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now()
    )
    updated_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime, onupdate=func.now()
    )

    def __repr__(self) -> str:
        return f"User(id={self.id}, username='{self.username}')"
```

### Model with Table Configuration

```python
class Product(Base):
    __tablename__ = "products"

    # Table arguments
    __table_args__ = (
        UniqueConstraint('name', 'category_id'),
        CheckConstraint('price > 0'),
        Index('idx_category_price', 'category_id', 'price'),
        {'mysql_engine': 'InnoDB', 'mysql_charset': 'utf8'}
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100))
    price: Mapped[Decimal] = mapped_column(Numeric(10, 2))
    category_id: Mapped[int] = mapped_column(ForeignKey("categories.id"))
```

---

## Column Types & Constraints

### Common Column Types

```python
from sqlalchemy import (
    String, Text, Integer, BigInteger, Float, Numeric,
    Boolean, Date, DateTime, Time, JSON, LargeBinary
)
from decimal import Decimal
import datetime

class Example(Base):
    __tablename__ = "examples"

    # Integers
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    big_id: Mapped[int] = mapped_column(BigInteger)

    # Strings
    name: Mapped[str] = mapped_column(String(50))
    description: Mapped[Optional[str]] = mapped_column(Text)

    # Numbers
    price: Mapped[Decimal] = mapped_column(Numeric(10, 2))
    rating: Mapped[float] = mapped_column(Float)

    # Boolean
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    # Dates and times
    birth_date: Mapped[date] = mapped_column(Date)
    created_at: Mapped[datetime] = mapped_column(DateTime)
    login_time: Mapped[time] = mapped_column(Time)

    # JSON (PostgreSQL, MySQL 5.7+, SQLite 3.38+)
    metadata: Mapped[dict] = mapped_column(JSON)

    # Binary data
    file_data: Mapped[bytes] = mapped_column(LargeBinary)
```

### Constraints and Indexes

```python
from sqlalchemy import CheckConstraint, UniqueConstraint, Index

class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)

    # Column-level constraints
    username: Mapped[str] = mapped_column(
        String(50),
        unique=True,
        nullable=False,
        index=True
    )

    age: Mapped[int] = mapped_column(
        Integer,
        CheckConstraint('age >= 0'),
        default=0
    )

    # Table-level constraints
    __table_args__ = (
        UniqueConstraint('email', 'username'),
        CheckConstraint('age BETWEEN 0 AND 150'),
        Index('idx_user_created', 'created_at'),
        Index('idx_user_multi', 'username', 'email'),
    )
```

---

## Relationships

### One-to-Many

```python
from typing import List
from sqlalchemy import ForeignKey
from sqlalchemy.orm import relationship

class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(String(50))

    # One user has many posts
    posts: Mapped[List["Post"]] = relationship(
        back_populates="author",
        cascade="all, delete-orphan"
    )

class Post(Base):
    __tablename__ = "posts"
    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(String(100))
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))

    # Many posts belong to one user
    author: Mapped["User"] = relationship(back_populates="posts")
```

### Many-to-Many

```python
from sqlalchemy import Table, Column

# Association table
post_tags = Table(
    "post_tags",
    Base.metadata,
    Column("post_id", ForeignKey("posts.id"), primary_key=True),
    Column("tag_id", ForeignKey("tags.id"), primary_key=True)
)

class Post(Base):
    __tablename__ = "posts"
    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(String(100))

    # Many-to-many with tags
    tags: Mapped[List["Tag"]] = relationship(
        secondary=post_tags,
        back_populates="posts"
    )

class Tag(Base):
    __tablename__ = "tags"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(50))

    posts: Mapped[List["Post"]] = relationship(
        secondary=post_tags,
        back_populates="tags"
    )
```

### Association Object Pattern

```python
class PostTag(Base):
    __tablename__ = "post_tags"

    post_id: Mapped[int] = mapped_column(ForeignKey("posts.id"), primary_key=True)
    tag_id: Mapped[int] = mapped_column(ForeignKey("tags.id"), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())

    # Relationships to parent objects
    post: Mapped["Post"] = relationship(back_populates="tag_associations")
    tag: Mapped["Tag"] = relationship(back_populates="post_associations")

class Post(Base):
    __tablename__ = "posts"
    id: Mapped[int] = mapped_column(primary_key=True)

    # Direct many-to-many (read-only)
    tags: Mapped[List["Tag"]] = relationship(
        secondary="post_tags",
        viewonly=True
    )

    # Association object access
    tag_associations: Mapped[List["PostTag"]] = relationship(
        back_populates="post"
    )
```

### Self-Referential

```python
class Category(Base):
    __tablename__ = "categories"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(50))
    parent_id: Mapped[Optional[int]] = mapped_column(ForeignKey("categories.id"))

    # Self-referential relationship
    parent: Mapped[Optional["Category"]] = relationship(
        "Category",
        remote_side=[id],
        back_populates="children"
    )
    children: Mapped[List["Category"]] = relationship(
        "Category",
        back_populates="parent"
    )
```

---

## Session Management

### Basic Session Usage

```python
from sqlalchemy.orm import sessionmaker, Session

# Create session factory
SessionLocal = sessionmaker(bind=engine)

# Context manager (recommended)
def get_user_by_id(user_id: int):
    with SessionLocal() as session:
        user = session.get(User, user_id)
        return user

# Manual session management
def create_user(username: str):
    session = SessionLocal()
    try:
        user = User(username=username)
        session.add(user)
        session.commit()
        return user
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
```

### Session Configuration

```python
# Session with custom configuration
SessionLocal = sessionmaker(
    bind=engine,
    expire_on_commit=False,  # Keep objects accessible after commit
    autoflush=True,          # Auto-flush before queries
    autocommit=False         # Manual transaction control
)

# Scoped sessions (thread-local)
from sqlalchemy.orm import scoped_session

SessionLocal = scoped_session(sessionmaker(bind=engine))

# Usage
session = SessionLocal()
# ... use session
SessionLocal.remove()  # Clean up
```

### Async Sessions

```python
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

# Async engine
async_engine = create_async_engine(
    "postgresql+asyncpg://user:pass@localhost/db",
    echo=True
)

# Async session factory
AsyncSessionLocal = async_sessionmaker(
    bind=async_engine,
    expire_on_commit=False
)

# Async usage
async def create_user_async(username: str):
    async with AsyncSessionLocal() as session:
        async with session.begin():
            user = User(username=username)
            session.add(user)
            # Auto-commit on exit
        return user
```

---

## CRUD Operations

### Create

```python
# Single object
def create_user(session: Session, username: str, email: str):
    user = User(username=username, email=email)
    session.add(user)
    session.commit()
    session.refresh(user)  # Get auto-generated fields
    return user

# Multiple objects
def create_users_bulk(session: Session, user_data: List[dict]):
    users = [User(**data) for data in user_data]
    session.add_all(users)
    session.commit()
    return users

# Bulk insert (more efficient)
def bulk_insert_users(session: Session, user_data: List[dict]):
    session.execute(
        insert(User).values(user_data)
    )
    session.commit()
```

### Read

```python
# Get by primary key
def get_user(session: Session, user_id: int):
    return session.get(User, user_id)

# Get first matching record
def get_user_by_username(session: Session, username: str):
    stmt = select(User).where(User.username == username)
    return session.scalar(stmt)

# Modern approach with select()
from sqlalchemy import select

def get_user_by_email(session: Session, email: str):
    stmt = select(User).where(User.email == email)
    return session.scalar(stmt)

# Get all with conditions
def get_active_users(session: Session):
    stmt = select(User).where(User.is_active == True)
    return session.scalars(stmt).all()

# Paginated results
def get_users_paginated(session: Session, page: int, per_page: int):
    stmt = (
        select(User)
        .offset((page - 1) * per_page)
        .limit(per_page)
    )
    return session.scalars(stmt).all()
```

### Update

```python
# Update single object
def update_user(session: Session, user_id: int, **kwargs):
    user = session.get(User, user_id)
    if user:
        for key, value in kwargs.items():
            setattr(user, key, value)
        session.commit()
        return user
    return None

# Bulk update
from sqlalchemy import update

def deactivate_old_users(session: Session, cutoff_date: date):
    stmt = (
        update(User)
        .where(User.last_login < cutoff_date)
        .values(is_active=False)
    )
    result = session.execute(stmt)
    session.commit()
    return result.rowcount
```

### Delete

```python
# Delete single object
def delete_user(session: Session, user_id: int):
    user = session.get(User, user_id)
    if user:
        session.delete(user)
        session.commit()
        return True
    return False

# Bulk delete
from sqlalchemy import delete

def delete_inactive_users(session: Session):
    stmt = delete(User).where(User.is_active == False)
    result = session.execute(stmt)
    session.commit()
    return result.rowcount
```

---

## Query Patterns

### Basic Filtering

```python
from sqlalchemy import select, and_, or_, not_

# Simple conditions
stmt = select(User).where(User.age > 18)
stmt = select(User).where(User.username.like('%john%'))
stmt = select(User).where(User.email.in_(['a@b.com', 'c@d.com']))

# Multiple conditions
stmt = select(User).where(
    and_(
        User.age >= 18,
        User.is_active == True,
        User.email.like('%@gmail.com')
    )
)

# OR conditions
stmt = select(User).where(
    or_(
        User.username == 'admin',
        User.role == 'superuser'
    )
)

# NOT conditions
stmt = select(User).where(not_(User.is_deleted))
```

### Ordering and Limiting

```python
# Order by
stmt = select(User).order_by(User.created_at.desc())
stmt = select(User).order_by(User.last_name, User.first_name)

# Limit and offset
stmt = select(User).limit(10).offset(20)

# Distinct
stmt = select(User.city).distinct()

# First, one, or scalar
user = session.scalars(stmt).first()  # First or None
user = session.scalars(stmt).one()    # Exactly one or error
users = session.scalars(stmt).all()   # All results
```

### Aggregations

```python
from sqlalchemy import func, desc

# Count
stmt = select(func.count(User.id))
total_users = session.scalar(stmt)

# Group by with aggregates
stmt = (
    select(User.city, func.count(User.id).label('user_count'))
    .group_by(User.city)
    .order_by(desc('user_count'))
)

# Having clause
stmt = (
    select(User.city, func.count(User.id).label('user_count'))
    .group_by(User.city)
    .having(func.count(User.id) > 5)
)
```

### Subqueries and CTEs

```python
# Subquery
subq = select(func.avg(User.age)).scalar_subquery()
stmt = select(User).where(User.age > subq)

# Correlated subquery
subq = (
    select(func.count(Post.id))
    .where(Post.user_id == User.id)
    .scalar_subquery()
)
stmt = select(User).where(subq > 5)

# Common Table Expression (CTE)
cte = select(User.city, func.count().label('cnt')).group_by(User.city).cte()
stmt = select(cte).where(cte.c.cnt > 10)
```

---

## Joins & Eager Loading

### Explicit Joins

```python
# Inner join
stmt = (
    select(User, Post)
    .join(Post, User.id == Post.user_id)
    .where(Post.published == True)
)

# Left outer join
stmt = (
    select(User)
    .outerjoin(Post)
    .where(Post.id.is_(None))  # Users with no posts
)

# Multiple joins
stmt = (
    select(User, Post, Category)
    .join(Post)
    .join(Category, Post.category_id == Category.id)
)
```

### Eager Loading

```python
from sqlalchemy.orm import selectinload, joinedload, contains_eager

# Select IN loading (separate query)
stmt = select(User).options(selectinload(User.posts))

# Joined loading (single query)
stmt = select(User).options(joinedload(User.posts))

# Nested relationships
stmt = select(User).options(
    selectinload(User.posts).selectinload(Post.comments)
)

# With explicit join
stmt = (
    select(User)
    .join(User.posts)
    .options(contains_eager(User.posts))
    .where(Post.published == True)
)
```

### Lazy Loading Control

```python
# Relationship configuration
class User(Base):
    posts: Mapped[List["Post"]] = relationship(
        lazy='selectin',  # Always use selectin loading
        # Other options: 'joined', 'subquery', 'raise', 'noload'
    )

# Query-level control
from sqlalchemy.orm import lazyload, noload, raiseload

stmt = select(User).options(
    lazyload(User.posts),     # Force lazy loading
    noload(User.profile),     # Don't load at all
    raiseload(User.settings)  # Raise error if accessed
)
```

---

## Advanced Querying

### Window Functions

```python
from sqlalchemy import func

# Row number
stmt = select(
    User.username,
    func.row_number().over(
        order_by=User.created_at
    ).label('row_num')
)

# Ranking
stmt = select(
    User.username,
    User.score,
    func.rank().over(
        order_by=User.score.desc()
    ).label('rank')
)

# Partition by
stmt = select(
    User.username,
    User.department,
    func.avg(User.salary).over(
        partition_by=User.department
    ).label('dept_avg_salary')
)
```

### Raw SQL and Text

```python
from sqlalchemy import text

# Raw SQL query
stmt = text("SELECT * FROM users WHERE age > :age")
result = session.execute(stmt, {'age': 18})

# Mixed SQL and ORM
stmt = select(User).where(text("age > :age")).params(age=18)

# Complex expressions
stmt = select(User).where(
    text("extract(year from created_at) = :year")
).params(year=2023)
```

### Dynamic Queries

```python
def build_user_query(session: Session, **filters):
    stmt = select(User)

    if name := filters.get('name'):
        stmt = stmt.where(User.name.like(f'%{name}%'))

    if min_age := filters.get('min_age'):
        stmt = stmt.where(User.age >= min_age)

    if cities := filters.get('cities'):
        stmt = stmt.where(User.city.in_(cities))

    if order_by := filters.get('order_by'):
        if order_by == 'name':
            stmt = stmt.order_by(User.name)
        elif order_by == 'age':
            stmt = stmt.order_by(User.age.desc())

    return session.scalars(stmt).all()
```

---

## Migrations with Alembic

### Setup

```bash
# Initialize Alembic
alembic init alembic

# Configure alembic.ini
sqlalchemy.url = sqlite:///database.db
```

```python
# alembic/env.py configuration
from myapp.models import Base

target_metadata = Base.metadata

def run_migrations_offline():
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"}
    )
    with context.begin_transaction():
        context.run_migrations()

def run_migrations_online():
    connectable = engine_from_config(
        config.get_section(config.config_ini_section),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata
        )
        with context.begin_transaction():
            context.run_migrations()
```

### Migration Commands

```bash
# Generate migration
alembic revision --autogenerate -m "Add user table"

# Apply migrations
alembic upgrade head
alembic upgrade +1        # Upgrade one revision
alembic upgrade ae10      # Upgrade to specific revision

# Downgrade
alembic downgrade -1      # Downgrade one revision
alembic downgrade base    # Downgrade to beginning

# Show current revision
alembic current

# Show migration history
alembic history

# Check if any new operations would be detected
alembic check
```

### Manual Migration Script

```python
"""Add user table

Revision ID: abc123
Revises: def456
Create Date: 2024-01-01 10:00:00.000000
"""

from alembic import op
import sqlalchemy as sa

# revision identifiers
revision = 'abc123'
down_revision = 'def456'
branch_labels = None
depends_on = None

def upgrade():
    op.create_table(
        'users',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('username', sa.String(50), nullable=False),
        sa.Column('email', sa.String(100), nullable=False),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now())
    )

    op.create_index('ix_users_username', 'users', ['username'])
    op.create_unique_constraint('uq_users_email', 'users', ['email'])

def downgrade():
    op.drop_constraint('uq_users_email', 'users', type_='unique')
    op.drop_index('ix_users_username', 'users')
    op.drop_table('users')
```

### Data Migrations

```python
from alembic import op
from sqlalchemy import text

def upgrade():
    # Schema change
    op.add_column('users', sa.Column('full_name', sa.String(100)))

    # Data migration
    connection = op.get_bind()
    connection.execute(
        text("UPDATE users SET full_name = first_name || ' ' || last_name")
    )

def downgrade():
    op.drop_column('users', 'full_name')
```

---

## Best Practices

### Session Management

```python
# ✅ Good: Use context managers
def get_user_posts(user_id: int):
    with SessionLocal() as session:
        stmt = (
            select(User)
            .options(selectinload(User.posts))
            .where(User.id == user_id)
        )
        return session.scalar(stmt)

# ✅ Good: Dependency injection pattern
def create_user(session: Session, username: str) -> User:
    user = User(username=username)
    session.add(user)
    session.commit()
    session.refresh(user)
    return user

# ❌ Bad: Long-lived sessions
global_session = SessionLocal()  # Don't do this
```

### Query Optimization

```python
# ✅ Good: Use selectinload for one-to-many
stmt = select(User).options(selectinload(User.posts))

# ✅ Good: Use joinedload for many-to-one
stmt = select(Post).options(joinedload(Post.author))

# ✅ Good: Load only needed columns
stmt = select(User.id, User.username).where(User.is_active == True)

# ✅ Good: Use bulk operations for large datasets
session.execute(
    update(User).where(User.last_login < cutoff).values(is_active=False)
)

# ❌ Bad: N+1 queries
users = session.scalars(select(User)).all()
for user in users:
    print(user.posts)  # Triggers separate query for each user
```

### Model Design

```python
# ✅ Good: Use type hints and proper defaults
class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(String(50), unique=True)
    is_active: Mapped[bool] = mapped_column(default=True)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    def __repr__(self) -> str:
        return f"User(id={self.id}, username='{self.username}')"

# ✅ Good: Use appropriate relationship loading
class User(Base):
    posts: Mapped[List["Post"]] = relationship(
        back_populates="author",
        lazy="selectin",  # Efficient for one-to-many
        cascade="all, delete-orphan"
    )

# ✅ Good: Index frequently queried columns
class Post(Base):
    __tablename__ = "posts"
    __table_args__ = (
        Index('idx_post_user_created', 'user_id', 'created_at'),
        Index('idx_post_status', 'status'),
    )
```

### Error Handling

```python
from sqlalchemy.exc import IntegrityError, NoResultFound

def create_user_safe(session: Session, username: str, email: str):
    try:
        user = User(username=username, email=email)
        session.add(user)
        session.commit()
        return user, None
    except IntegrityError as e:
        session.rollback()
        if 'username' in str(e):
            return None, "Username already exists"
        elif 'email' in str(e):
            return None, "Email already exists"
        return None, "Database constraint error"
    except Exception as e:
        session.rollback()
        return None, f"Unexpected error: {str(e)}"

def get_user_or_404(session: Session, user_id: int):
    user = session.get(User, user_id)
    if not user:
        raise NoResultFound(f"User {user_id} not found")
    return user
```

### Testing Patterns

```python
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

@pytest.fixture
def db_session():
    # Use in-memory SQLite for tests
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)

    session = Session()
    try:
        yield session
    finally:
        session.close()

def test_create_user(db_session):
    user = User(username="testuser", email="test@example.com")
    db_session.add(user)
    db_session.commit()

    assert user.id is not None
    assert user.username == "testuser"

    # Test retrieval
    retrieved = db_session.get(User, user.id)
    assert retrieved.username == "testuser"
```

---

## Common Gotchas

- **Expired objects after commit**: Use `expire_on_commit=False` or `session.refresh(obj)`
- **N+1 queries**: Use eager loading (`selectinload`, `joinedload`)
- **Session scope**: Always use context managers or proper cleanup
- **Timezone-aware datetimes**: Use `timezone=True` for DateTime columns
- **Bulk operations**: Use `bulk_insert_mappings()` for large datasets
- **Connection pooling**: Configure pool settings for production
- **Migration conflicts**: Always review auto-generated migrations
- **Relationship loading**: Choose appropriate `lazy` loading strategies

This cheat sheet covers the essential SQLAlchemy patterns for modern Python applications. For async usage, replace `Session` with `AsyncSession` and add `async`/`await` keywords as needed.