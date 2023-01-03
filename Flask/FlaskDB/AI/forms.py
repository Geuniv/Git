from flask_wtf import FlaskForm
from wtforms import StringField, TextAreaField , PasswordField , EmailField
from wtforms.validators import DataRequired , Length , EqualTo , Email

class UserCreateForm(FlaskForm):
    username = StringField('사용자이름', validators=[DataRequired("ID를 확인해주세요"), Length(min=3, max=25)]) # Length 오류 메세지 찾는중 + 이메일 / 폰 예외처리
    password1 = PasswordField('비밀번호', validators=[DataRequired("비밀번호를 확인해주세요"), Length(min=8, max=20) , EqualTo('password2', '비밀번호가 다릅니다.')])
    password2 = PasswordField('비밀번호 확인', validators=[DataRequired() , Length(min=8, max=20)])
    name = StringField('이름' , validators=[DataRequired("이름을 확인해주세요") , Length(min=3 , max=20)])
    birthday = StringField('생년월일' , validators=[DataRequired("생년월일을 확인해주세요") , Length(min=8 , max=8)])
    gender = StringField("성별" , validators=[DataRequired("성별을 확인해주세요") , Length(min=1 , max=8)])
    address1 = StringField('주소' , validators=[DataRequired("주소를 확인해주세요")])
    address2 = StringField('상세주소' , validators=[DataRequired("상세주소를 확인해주세요")])
    email = EmailField('이메일', validators=[DataRequired("이메일을 확인해주세요"), Email()])
    phone = StringField('전화번호', validators=[DataRequired("전화번호를 확인해주세요")])

class UserLoginForm(FlaskForm):
    username = StringField('유저 이름' , validators=[DataRequired('ID를 확인해주세요'), Length(min=3 , max=25)])
    password = PasswordField('비밀번호' , validators=[DataRequired('비밀번호를 확인해주세요')])

class QuestionForm(FlaskForm):
    subject = StringField('제목', validators=[DataRequired('제목은 필수입력 항목입니다.')])        # 한줄만 넣을 수 있음
    content = TextAreaField('내용', validators=[DataRequired('내용은 필수입력 항목입니다.')])     # 여러줄을 넣을 수 있음

class AnswerForm(FlaskForm):
    content = TextAreaField('내용', validators=[DataRequired('내용은 필수입력 항목입니다.')])