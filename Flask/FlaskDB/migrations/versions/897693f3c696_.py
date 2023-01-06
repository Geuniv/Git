"""empty message

Revision ID: 897693f3c696
Revises: e98fd787419e
Create Date: 2023-01-06 16:57:34.094228

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import mysql

# revision identifiers, used by Alembic.
revision = '897693f3c696'
down_revision = 'e98fd787419e'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('test',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('user_id', sa.Integer(), server_default='1', nullable=True),
    sa.Column('cataract', sa.String(length=100), nullable=False),
    sa.Column('accuracy', sa.Integer(), nullable=False),
    sa.Column('run_date', sa.DateTime(), nullable=False),
    sa.ForeignKeyConstraint(['user_id'], ['user.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )
    with op.batch_alter_table('hospital_info', schema=None) as batch_op:
        batch_op.drop_index('ix_hospital_info_index')

    op.drop_table('hospital_info')
    with op.batch_alter_table('answer', schema=None) as batch_op:
        batch_op.create_foreign_key(None, 'user', ['user_id'], ['id'], ondelete='CASCADE')

    with op.batch_alter_table('question', schema=None) as batch_op:
        batch_op.create_foreign_key(None, 'user', ['user_id'], ['id'], ondelete='CASCADE')

    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table('question', schema=None) as batch_op:
        batch_op.drop_constraint(None, type_='foreignkey')

    with op.batch_alter_table('answer', schema=None) as batch_op:
        batch_op.drop_constraint(None, type_='foreignkey')

    op.create_table('hospital_info',
    sa.Column('index', mysql.BIGINT(), autoincrement=False, nullable=True),
    sa.Column('dutyaddr', mysql.TEXT(), nullable=True),
    sa.Column('dutydiv', mysql.TEXT(), nullable=True),
    sa.Column('dutydivnam', mysql.TEXT(), nullable=True),
    sa.Column('dutyemcls', mysql.TEXT(), nullable=True),
    sa.Column('dutyemclsname', mysql.TEXT(), nullable=True),
    sa.Column('dutyeryn', mysql.TEXT(), nullable=True),
    sa.Column('dutyetc', mysql.TEXT(), nullable=True),
    sa.Column('dutymapimg', mysql.TEXT(), nullable=True),
    sa.Column('dutyname', mysql.TEXT(), nullable=True),
    sa.Column('dutytel1', mysql.TEXT(), nullable=True),
    sa.Column('dutytel3', mysql.TEXT(), nullable=True),
    sa.Column('dutytime1c', mysql.TEXT(), nullable=True),
    sa.Column('dutytime2c', mysql.TEXT(), nullable=True),
    sa.Column('dutytime3c', mysql.TEXT(), nullable=True),
    sa.Column('dutytime4c', mysql.TEXT(), nullable=True),
    sa.Column('dutytime5c', mysql.TEXT(), nullable=True),
    sa.Column('dutytime6c', mysql.TEXT(), nullable=True),
    sa.Column('dutytime7c', mysql.TEXT(), nullable=True),
    sa.Column('dutytime8c', mysql.TEXT(), nullable=True),
    sa.Column('dutytime1s', mysql.TEXT(), nullable=True),
    sa.Column('dutytime2s', mysql.TEXT(), nullable=True),
    sa.Column('dutytime3s', mysql.TEXT(), nullable=True),
    sa.Column('dutytime4s', mysql.TEXT(), nullable=True),
    sa.Column('dutytime5s', mysql.TEXT(), nullable=True),
    sa.Column('dutytime6s', mysql.TEXT(), nullable=True),
    sa.Column('dutytime7s', mysql.TEXT(), nullable=True),
    sa.Column('dutytime8s', mysql.TEXT(), nullable=True),
    sa.Column('hpid', mysql.TEXT(), nullable=True),
    sa.Column('postcdn1', mysql.TEXT(), nullable=True),
    sa.Column('postcdn2', mysql.TEXT(), nullable=True),
    sa.Column('wgs84lon', mysql.TEXT(), nullable=True),
    sa.Column('wgs84lat', mysql.TEXT(), nullable=True),
    sa.Column('dutyinf', mysql.TEXT(), nullable=True),
    mysql_default_charset='utf8mb3',
    mysql_engine='InnoDB'
    )
    with op.batch_alter_table('hospital_info', schema=None) as batch_op:
        batch_op.create_index('ix_hospital_info_index', ['index'], unique=False)

    op.drop_table('test')
    # ### end Alembic commands ###
