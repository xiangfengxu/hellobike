import zmail
from typing import List


class MyEmail:

    @staticmethod
    def send_mail(subject: str, content_text: str, to_addr: List[str] or str) -> bool:

        # 邮件的主题和内容
        mail_content = {'subject': subject, 'content_text': content_text}
        try:
            # 配置发送方的邮箱和密码
            server = zmail.server('buddaa@163.com', 'xdyxxf99')
            return server.send_mail(to_addr, mail_content)
        except:
            return False


if __name__ == '__main__':
    is_success = MyEmail.send_mail("般若波罗蜜多心经",
                                   "观自在菩萨，行深般若波罗蜜多时，照见......",
                                   'buddaa@foxmail.com')
    if is_success:
        print("send email success")
    else:
        print("send email failed")
