import psycopg2

def open_connection(dbname,user,password,host,port='5432'):
    """Connect to a PostgreSQL database 
    
    Arguments:
        dbname: String, database name
        user: String, database username
        password: String, database password for username
        host: String, IP Address trying to connect to
        port: String, should be 5432 by default
        
    Returns: Dictionary, with connection and cursor"""

    try:
        connection = psycopg2.connect(
            dbname=dbname,
            user=user,
            password=password,
            host=host,
            port=port 
        )
        cursor = connection.cursor()

    except (Exception, psycopg2.Error) as error:
        print("Error while connecting to PostgreSQL:", error)
    
    return {'connection': connection, 'cursor': cursor} 

def run_query(cursor,sql_statement):
    """Run an SQL statement and retrieve results from a PSQL database
    
    Arguments:
        cursor: Connection to PSQL Database
        sql_statement: String, SQL Statement
        
    Returns: List of rows, from psycopg2"""

    cursor.execute(sql_statement)
    column_names = [desc[0] for desc in cursor.description]
    
    results = []
    for row in cursor.fetchall():
        row_dict = dict(zip(column_names, row))
        results.append(row_dict)

    return  results 

def close_connection(connection, cursor):
    """Close connection to PSQL database
    
    Arguments:
        conection: Connection object from PSQL
        cusor: Cursor from the connection
        
    Returns: Nothing
    
    Side Effects: Closes both the connection and cursor"""

    cursor.close()
    connection.close()
