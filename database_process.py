import os  # For file operations
import sqlite3  # For interacting with SQLite databases

def set_sql_connect(database_name):
    """
    Establishes a connection to the SQLite database.

    Parameters:
    - database_name (str): The name of the database file.

    Returns:
    - sqlite3.Connection: A connection object to interact with the database.
    """
    return sqlite3.connect(database_name)

def set_sql_cursor(database_connect):
    """
    Creates a cursor object for executing SQL commands.

    Parameters:
    - database_connect (sqlite3.Connection): The database connection.

    Returns:
    - sqlite3.Cursor: A cursor object for executing queries.
    """
    return database_connect.cursor()

def close_connect(vt):
    """
    Commits any changes and closes the database connection.

    Parameters:
    - vt (sqlite3.Connection): The active database connection.
    """
    if vt:
        vt.commit()  # Save any changes made during the session
        vt.close()  # Close the connection

def set_connect_and_cursor(path='Data/database.sqlite'):
    """
    Establishes a connection to the database and creates a cursor.

    Parameters:
    - path (str, optional): The path to the SQLite database file. Defaults to 'Data/database.sqlite'.

    Returns:
    - tuple: (sqlite3.Connection, sqlite3.Cursor) for further database operations.
    """
    vt = set_sql_connect(path)  # Connect to the database
    db = set_sql_cursor(vt)  # Create a cursor
    return vt, db  # Return connection and cursor for further queries

def create_table(table_name, columns):
    """
    Creates a table in the database if it does not already exist.

    Parameters:
    - table_name (str): The name of the table.
    - columns (str): The column definitions in SQL format (e.g., "id INTEGER PRIMARY KEY, name TEXT").
    """
    vt, db = set_connect_and_cursor()  # Establish database connection and cursor
    db.execute("CREATE TABLE IF NOT EXISTS {0} ({1})".format(table_name, columns))  # Execute table creation command
    close_connect(vt)  # Commit and close the connection

def get_data(sql_command):
    """
    Retrieves data from the database based on the provided SQL command.

    Parameters:
    - sql_command (str): The SQL query to execute.

    Returns:
    - list: A list of tuples containing the retrieved data.
    """
    vt, db = set_connect_and_cursor()  # Establish database connection and cursor
    db.execute(sql_command)  # Execute the provided SQL command
    gelen_veri = db.fetchall()  # Fetch all results
    close_connect(vt)  # Commit and close the connection
    return gelen_veri  # Return the fetched data

def add_data(table, adding):
    """
    Inserts data into the specified table.

    Parameters:
    - table (str): The table name.
    - adding (str): The values to insert, formatted as an SQL values string (e.g., "'John', 25").
    """
    vt, db = set_connect_and_cursor()  # Establish database connection and cursor
    db.execute("INSERT INTO '{0}' VALUES ({1})".format(table, adding))  # Execute the insert command
    close_connect(vt)  # Commit and close the connection
