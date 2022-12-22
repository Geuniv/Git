from AI import db

# PK값은 무조건 있어야 함.

class User(db.Model):
    id = db.Column(db.Integer , primary_key = True )
    username = db.Column(db.String(30) , unique = True, nullable = False)
    password = db.Column(db.String(200) , nullable = False)
    name = db.Column(db.String(10) , nullable = False)
    birthday = db.Column(db.Integer , nullable = False)
    gender = db.Column(db.String(10) , nullable = False)
    address1 = db.Column(db.String(120) , nullable = False)
    address2 = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(120) , unique = True, nullable = False)
    phone = db.Column(db.String(30) , unique = True, nullable = False)
    reg_date = db.Column(db.DateTime() , nullable = False)