from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

c = canvas.Canvas("SentenialX_Full_CheatSheet.pdf", pagesize=letter)
width, height = letter

# Title
c.setFont("Helvetica-Bold", 20)
c.drawString(50, height - 50, "Sentenial-X Full CLI & API Cheat Sheet")

# Embed diagrams
c.drawImage("assets/diagrams/architecture.png", 50, height - 350, width=500, height=250)
c.drawImage("assets/diagrams/data_flow.png", 50, height - 650, width=500, height=250)

c.save()
