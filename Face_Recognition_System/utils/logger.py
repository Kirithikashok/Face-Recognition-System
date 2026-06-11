import sqlite3


def save_log(name, confidence):

    conn = sqlite3.connect("database.db")

    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO logs(name,confidence)
        VALUES (?,?)
        """,
        (name, confidence)
    )

    conn.commit()
    conn.close()