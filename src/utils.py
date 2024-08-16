from typing import Dict, Tuple
import mysql.connector


def get_umls_data(cui: str, connector: mysql.connector) -> Dict[str, Tuple[str]]:
    data = {'UMLS_NAME': None, 'MODE': None, 'MAX': None, 'MIN': None, 'DEF': None}
    cursor = connector.cursor(buffered=True)
    query = (f"SELECT STR FROM MRCONSO WHERE CUI='{cui}' AND LAT='ENG' AND ISPREF = 'Y' GROUP BY STR")
    cursor.execute(query)

    # gets the UMLS name if it has it
    try:
        temp = cursor.fetchone()
        if temp is not None and temp != '':
            data['UMLS_NAME'] = temp
        else:
            data['UMLS_NAME'] = ('',)
    except:
        data['UMLS_NAME'] = ('',)

    # returns the mode of the names if it has one
    query = (f"SELECT STR \
            FROM MRCONSO WHERE CUI='{cui}' AND LAT='ENG' \
            GROUP BY STR ORDER BY COUNT(LENGTH(STR)) DESC;")
    cursor.execute(query)
    try:
        temp = cursor.fetchone()
        if temp is not None and temp != '':
            data['MODE'] = temp
        else:
            data['MODE'] = ('',)
    except:
        data['MODE'] = ('',)

    # returns the longest length name in UMLS if it has one
    query = (f"SELECT STR FROM MRCONSO \
        WHERE CUI='{cui}' AND LAT='ENG' \
        AND LENGTH(STR) = \
        (SELECT MAX(LENGTH(STR)) FROM MRCONSO \
        WHERE CUI='{cui}' AND LAT='ENG')")
    cursor.execute(query)
    try:
        temp = cursor.fetchone()
        if temp is not None and temp != '':
            data['MAX'] = temp
        else:
            data['MAX'] = ('',)
    except:
        data['MAX'] = ('',)

    # returns the shortest length name in UMLS if it has one
    query = (f"SELECT STR FROM MRCONSO \
        WHERE CUI='{cui}' AND LAT='ENG' \
        AND LENGTH(STR) = \
        (SELECT MIN(LENGTH(STR)) FROM MRCONSO \
        WHERE CUI='{cui}' AND LAT='ENG')")
    cursor.execute(query)
    try:
        temp = cursor.fetchone()
        if temp is not None and temp != '':
            data['MIN'] = temp
        else:
            data['MIN'] = ('',)
    except:
        data['MIN'] = ('',)
    
    # returns the definition in UMLS if it has one
    query = (f"SELECT DEF FROM MRDEF WHERE CUI='{cui}' LIMIT 1")
    cursor.execute(query)
    try:
        temp = cursor.fetchone()
        if temp is not None and temp != '':
            data['DEF'] = temp
        else:
            data['DEF'] = ('',)
    except:
        data['DEF'] = ('',)
    cursor.close()
    return data
