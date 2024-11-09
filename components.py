import streamlit as st
from dataclasses import dataclass

@dataclass
class Statement:
    text: str
    votes: int = 0
    id: str = ""

class StatementDisplay:
    def render_statement_card(self, statement: Statement, key: str = None):
        """Render a statement card with voting button"""
        col1, col2 = st.columns([4, 1])
        with col1:
            st.write(statement.text)
        with col2:
            if st.button("⬆️ Upvote", key=f"upvote_{key}" if key else None):
                statement.votes += 1
                return True
        return False
